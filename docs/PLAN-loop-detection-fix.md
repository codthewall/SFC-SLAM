# 回环检测与位姿图（Loop / PGO）：诊断与修复计划

> 范围：`src/sfc_core/src/loopdetection/`（`loop_detector.cpp` 1321 行、`pose_graph_optimizer.cpp` 1497 行、`geometric_verifier.hpp` 1103 行、`sim3_solver.cpp` 422 行、`sim3_optimizer.cpp` 336 行、`place_recognition.hpp` 973 行）与 `Tracking::detectAndCorrectLoop()` / `applyOptimizedPosesToFrontend()`。
>
> 前置：前端 R1–R5（见 `docs/PLAN-frontend-pose-fix.md`）未修好之前，回环层**无论怎么改都不会收敛**——在错误的位姿序列上做地点识别，找到的都是错的候选。本文件只解决"回环层自身即使拿到正确输入也不工作"的问题。

---

## 一、结论摘要

回环层的问题不是"阈值调得不好"，是**唯一有真实尺度的验证路径从未被执行，而实际执行的两条路径一个尺度是假的、一个匹配关系是假的；同时 PGO 把里程计约束全部排除在优化之外，导致每次"回环成功"实际只解了一条边**。

| 编号 | 问题 | 后果 | 代码位置 |
|---|---|---|---|
| **L1** | Sim3 输入点被**两次** world→camera 变换 | 即使走到 Sim3，解出的相对位姿不是任何物理量 | `loop_detector.cpp:472-490` vs `:1163-1208` |
| **L2** | `detect()` 第 4 参写死 `nullptr` → `covis_matches` 恒空 → Sim3 分支永不进入 | 唯一有真实尺度的路径被彻底旁路 | `tracking.cpp:3352-3353`、`loop_detector.cpp:348`、`:472` |
| **L3** | PGO 里程计边 `setLevel(1)`，而 `optimize()` 调 `initializeOptimization()`（默认 level 0） | 优化时图里只剩 1 条回环边 → `chi2` 恒 0 → 什么都没纠正却计为成功 | `pose_graph_optimizer.cpp:272`、`:1316` |
| **L4** | `addLoopEdge()` 返回 `void`，去重命中后静默 `return`，调用方无法区分 | 43 次"回环"里绝大多数没有新增任何约束 | `pose_graph_optimizer.cpp:772-831`、`tracking.cpp:3365-3395` |
| **L5** | `geometric_verifier` 阈值写死在 `LoopDetector` 构造函数里，`config/geometric_verifier.yaml` 整份无消费者 | yaml / 结构体默认 / 写死值三套并存且写死的最松：`min_icp_inlier_ratio` 0.20/0.20/**0.10**、`icp_reproj_thresh_px` 25.0/20.0/**35.0** | `loop_detector.cpp:155-172` |
| **L6** | ICP 路径内点率分母用错（除 `matches.size()`，而 ICP 只处理过滤后的 `valid_matches`） | 内点率失真，`max_icp_inlier_ratio` 退化判据失效 | `loop_detector.cpp:535` |
| **L7** | ICP 用 `dummy_matches`（索引一一对应），完全不看描述子 | 3D-3D 匹配关系可能整体错位，旋转可差 38.6°（注释自承） | `loop_detector.cpp:524-527` |
| **L8** | 120 行崩溃诊断代码整块注释 + `sigsetjmp` 捕获 SIGSEGV 后继续运行 | 掩盖真实内存错误；可读性债务 | `pose_graph_optimizer.cpp:1196-1315`、`:1325-1421` |
| **L9** | `correctLoopWithScale()` / `computeCandidateConsistency()` 全仓无调用点 | 尺度估计与候选一致性检查都是死代码 | `loop_detector.cpp:908-957`、`:1113+` |
| **L10** | `geometric_verifier.hpp` **45 处 `std::cerr`，0 处守卫**（结构体里连 `verbose` 字段都没有） | 每次几何验证刷几十行，stderr 吞掉性能；R-017/R-021 违规 | `geometric_verifier.hpp` 全文 |
| **L11** | `selectBestFromVerified()` 的排序 lambda 不是严格弱序 | `std::sort` 行为未定义，候选选择不可复现 | `loop_detector.cpp:1062-1070` |

---

## 二、实测基线（fr1_desk2，640 帧，`--slam`）

| 指标 | 纯 VO | 完整 SLAM | 真值参考 |
|---|---|---|---|
| 路径总长 | 478.569 m | **1165.124 m** | 约 5–12 m |
| 起终点漂移 | 4.300 m | 1.341 m | < 0.3 m |
| 关键帧数 | 32 | 73 | — |
| 平均关键帧间隔 | 1.7957 m | 1.9577 m | 0.03–0.07 m |
| 单帧耗时 | 约 50 ms | **约 1 s** | — |
| 有效回环边 | — | **1 / 43 次** | — |

关键日志证据：

```
[PGO] optimize: vertices=65 edges=64 odom_edge_data_=63 loop_edge_data_=1
iteration= N chi2= 0.0
lambda= 2485.7 / 603298.6 / 70619.3
[PGO] addLoopEdge 跳过（loop_edge_ids_中已存在）: 623 -> 621
[Loop] 回环修正完成：KF=623 ↔ 621 iterations=16 修正关键帧=65 累计回环=35
[GeoVerifier] ICP inliers=62/946 (6.6%) |t|=0.1 m → 内点不足，迭代 SVD 回退也失败
[GeoVerifier] Essential: inliers=1108/1214 (91.3%) PASS
[GeoVerifier] Essential: inliers=86/549 (15.7%) PASS
[GeoVerifier] Relative pose: |t|=1.0 m          ← 每一次都是 1.0
```

判读：
- `odom_edge_data_=63` 但 `loop_edge_data_=1` 且 `chi2` 每轮 0.0 —— 图的约束不足以让任何顶点移动（L3）。
- `|t|=1.0` 每次恒定 —— Essential 只给方向，代码用 `kEMFallbackTranslationScale = 1.0` 顶替尺度（`loop_detector.cpp:631`、`:636`）。
- `15.7% PASS` —— `min_inlier_ratio` 写死 0.15，15.x% 刚好擦线（L5）。
- ICP 与 Essential 的判定完全脱节：同一对候选，ICP 6.6% 失败、Essential 91.3% 通过，最终采用后者。

---

## 三、根因逐条

### L1（P0）Sim3 输入被两次 world→camera 变换

调用侧 `loop_detector.cpp:472-490` 已经把世界点转到相机系：

```cpp
for (const auto& cm : covis_matches) {
    pts_curr.push_back(R_cw_curr * cm.pt_world + t_cw_curr);      // 已在相机系
    pts_matched.push_back(R_cw_match * cm.pt_world + t_cw_match);
    ...
}
geometricVerificationWithSim3(pts_curr, ..., pts_matched, ...);
```

被调函数 `:1163-1208` **又做一次**同样变换（形参名仍是 `*_pts3d_world`）：

```cpp
for (int i = 0; i < N; ++i) {
    Eigen::Vector3d pc1 = R_cw_curr * curr_pts3d_world[i] + t_cw_curr;
    Eigen::Vector3d pc2 = R_cw_matched * matched_pts3d_world[i] + t_cw_matched;
    ...
    pts_cam_curr.push_back(R_cw_curr * curr_pts3d_world[i] + t_cw_curr);   // 第二次
    pts_cam_matched.push_back(R_cw_matched * matched_pts3d_world[i] + t_cw_matched);
}
```

送进 `Sim3Solver` 的两组点各自被额外旋转了一次，且两个额外旋转不同（`R_cw_curr ≠ R_cw_matched`）。Sim3 求解的"两帧相机系点云对齐"变成对齐两份被不同旋转污染的点云 → 解出的 R/t 既不是 `T_12` 也不是任何物理量。

附带：`:1237 int solver_n = pts_cam_curr.size();` 与 `:1249 const Eigen::Vector3d& pw = curr_pts3d_world[i];` 混用两套数组长度。当 `use_filtered == true`（`:1181`，有点被过滤）时 `solver_n > curr_pts3d_world.size()` 在语义上的对应关系已错位。

**修法**：`geometricVerificationWithSim3` 只接受世界点，内部统一做一次变换；或在调用侧只传世界点、删掉 `:476-477` 的预变换。两者择一，不能都保留。

---

### L2（P0）`detect()` 永远收不到 `covis_mgr` → Sim3 分支永不执行

```cpp
// src/sfc_core/src/frontend/tracking.cpp:3352
const auto candidate = loop_detector_->detect(kf.keypoints, kf.pts_3d,
                                              kf.descriptors, nullptr);
```

`loop_detector.cpp:348 if (covis_mgr != nullptr) { ... }` 整段（含阶段 1 共视点匹配、阶段 2 描述子 fallback、`:472` 的 Sim3 调用）全部跳过 → `covis_matches` 恒空。

连锁：
- `:514 bool skip_icp_dummy = (!covis_matches.empty() && covis_matches.size() < 3);` → 恒 `false`
- → ICP 用它那套 `dummy_matches`（`:524-527`，索引一一对应）跑
- → ICP 失败后必落到 Essential（`:605-661`，尺度固定 1.0 m）

**根因链**：`Tracking` 没有 `CovisManager` 实例（`tracking.hpp:149 getCovisManager() const { return nullptr; }`），B2 接线时无从传入；而 `LoopDetector` 的 Sim3 路径在设计上完全依赖 `CovisManager::getAllConfirmedMapPoints()`。

**修法（二选一）**：
- **A（推荐，改动小）**：Sim3 的输入不依赖 CovisManager，改用 `keyframe_db_` 里两帧各自的 `entry.pts3d`（相机系点）+ 描述子匹配 `matches` 构造对应点对。这条路不需要新增任何组件，且输入是"这条 KF 自己的观测"，与 CovisManager 的 confirmed map point 在语义上等价（后者只是多了一层全局优化）。
- **B**：给 `Tracking` 建一个最小 CovisManager（只用来提供 confirmed map points）。代价：多一条数据流、多一套生命周期要维护，且 CovisManager 的 target 目前是注释状态。

无论哪条，都必须先把 L1 的双重变换修掉，否则 A 也没有意义（A 传的是相机系点，正好踩中同一个坑）。

---

### L3（P0）PGO 把里程计边设为 level 1，优化时只剩回环边

`pose_graph_optimizer.cpp:263-272`（`rebuildOptimizer` 的 Step 3）：

```cpp
auto* edge = new g2o::EdgeSE3Expmap();
...
edge->information() = data.information;
// ★★★ 里程计边设为 level=1，仅 loop 边参与优化 ★★★
// setLevel(0) 会让 odom 边参与 PGO，信息矩阵不准确时会扭曲整个轨迹
// 保持 level=1，让 PGO 只靠回环边修正，这是更安全的默认行为
edge->setLevel(1);
```

而 `optimize()` 在 `:1316`：

```cpp
int init = optimizer_->initializeOptimization();   // 默认 level = 0
```

g2o 的 `initializeOptimization(int level = 0)` 只激活 `edge->level() <= level` 的边。于是：

- 里程计边（level 1）**全部不参与**
- 图里只剩 loop 边（Step 4，`:311-319`，**没有** `setLevel`，默认 0）

结果完全对应实测：`edges=64`、`loop_edge_data_=1` → 实际参与优化的边只有 1 条 → 两端顶点的残差在设计上就该被推到 0 → `chi2 = 0.0` → Levenberg 的 `lambda` 只能一路升到 6e5 → "最大修正"仅 0.08/0.02/0.003 m。

**这不叫"更安全"，这叫"不优化"。** 回环的真正作用是"用回环边把闭环误差摊到路径上的所有里程计边上"，把里程计边排除就等于把闭环误差全部压在两个顶点上，其余 63 个顶点一个都不动。

同一文件里还有第二处不一致：`addOdometryEdge()`（`:611-728`）**没有** `setLevel(1)`，只设置 `edge->information()`（`:687-698`）。所以新建的 odom 边是 level 0，只有彻底重建后加的才变 level 1 —— 同一类边在同一优化器里 level 不一致，行为取决于是否触发过重建。

**修法**：
1. 删除 `:272` 的 `edge->setLevel(1)`。
2. `initializeOptimization()` 显式传 level（`initializeOptimization(0)`），并在注释里写明"level 0 = 全部边参与"。
3. 里程计边的可信度用 `information()` 表达，不要用 level 表达。`addOdometryEdge()` 已经有 `info_scale` 与钳位逻辑（`:671-698`），这条路是通的。
4. 保留 loop 边上的 Huber（`:317-319`），这才是"抑制错误回环"的正确手段。

---

### L4（P1）`addLoopEdge` 去重后无法回传失败

```cpp
// pose_graph_optimizer.hpp:80
void addLoopEdge(int kf_id_from, int kf_id_to, ...);
```

`pose_graph_optimizer.cpp:772-797`（`loop_edge_key_set_` 命中）与 `:798-831`（`loop_edge_ids_` 兜底）都是打印 `[PGO] addLoopEdge 跳过...` 后 `return;`。返回类型 `void`，调用方无从得知。

`tracking.cpp:3365-3395`：

```cpp
pose_graph_optimizer_->addLoopEdge(...);        // 可能是"跳过"
int iter = 0;
iter = pose_graph_optimizer_->optimize();       // 无条件
if (iter <= 0) return false;
const int n_corrected = applyOptimizedPosesToFrontend();
if (n_corrected <= 0) return false;
mnDetectedLoops_++;                              // 计为成功
```

**修法**：`addLoopEdge` 改返回 `bool`（true = 新边已加入，false = 重复被跳过/顶点缺失/几何非法）。`detectAndCorrectLoop()` 拿到 `false` 时直接 `return false`，不再调 `optimize()`。这样 `mnDetectedLoops_` 才是真实计数。

---

### L5（P1）`geometric_verifier` 配置写死，yaml 整份无消费者

`loop_detector.cpp:155-172`：

```cpp
GeometricVerifierConfig geo_config;
geo_config.ransac_iterations = config_.ransac_iterations;        // 来自 LoopDetectorConfig
geo_config.threshold_px = config_.ransac_threshold_px;           // 来自 LoopDetectorConfig
geo_config.min_inlier_ratio = 0.15;      // 写死
geo_config.min_inliers = 2;              // 写死
geo_config.min_icp_inliers = 2;          // 写死
geo_config.min_icp_inlier_ratio = 0.10;  // 写死
geo_config.icp_reproj_thresh_px = 35.0;  // 写死
geo_config.max_icp_inlier_ratio = config_.max_icp_inlier_ratio;  // 来自 LoopDetectorConfig
geo_config.min_depth = 0.1;              // 写死
geo_config.max_depth = 12.0;             // 写死
```

对比 `config/geometric_verifier.yaml`：

| 字段 | yaml | 结构体默认值 | 实际生效（写死） | 差异 |
|---|---|---|---|---|
| `min_inlier_ratio` | 0.15 | 0.20 | 0.15 | 恰好等于 yaml，比结构体默认更松 |
| `min_inliers` | 2 | 4 | 2 | 比结构体默认松一倍 |
| `min_icp_inliers` | 2 | **4** | **2** | 比结构体默认松一倍 |
| `min_icp_inlier_ratio` | **0.20** | **0.20** | **0.10** | 比 yaml 与默认值都松一倍 |
| `icp_reproj_thresh_px` | **25.0** | **20.0** | **35.0** | 比 yaml 松 40%，比默认值松 75% |

（结构体默认值见 `geometric_verifier.hpp:27-52`：`min_inlier_ratio=0.20`、`min_inliers=4`、`min_icp_inliers=4`、`min_icp_inlier_ratio=0.20`、`icp_reproj_thresh_px=20.0`、`max_icp_inlier_ratio=0.98`、`min_depth=0.1`、`max_depth=12.0`。）

**三套值互相矛盾，且写死的那套最松。** 更值得注意的是 `geometric_verifier.hpp` 里这些默认值本身带着"从 X 放宽到 Y""从 2 提高到 4"的调参注释，说明历史上已经在这三套值之间来回改过多次，但从未让配置文件生效。

`loadGeometricVerifierConfig`（`src/sfc_core/src/config/config_loader.cpp:93-125`）确实把它读进 `AllConfigs::geometric_verifier`（`:624-625`），但**全仓没有第二个消费者**——`Tracking` 构造 `LoopDetector` 时只传 `LoopDetectorConfig`。R-019/R-022/R-023 三重违规。

**修法**：把 `GeometricVerifierConfig` 作为 `LoopDetectorConfig` 的成员（`LoopDetectorConfig::geometric_verifier`），由 `loadAllConfigs()` 在唯一合并点注入；`LoopDetector` 构造函数里只保留"确实属于 LoopDetector 自己"的两项（`ransac_iterations`、`threshold_px`）的派生注释，其余一律读配置。

---

### L6（P1）ICP 内点率分母用错

`loop_detector.cpp:529-535`：

```cpp
int icp_inliers = geometric_verifier_->verifyWithICP3D(
    pts3d_query_cam_valid, pts3d_train_cam_valid, dummy_matches, K_cv_, &T_icp);
if (icp_inliers > 0) {
    double inlier_ratio = static_cast<double>(icp_inliers) / matches.size();  // ← 分母错
```

`dummy_matches` 的规模是 `pts3d_query_cam_valid.size()`（`:525`），而 `matches.size()` 是描述子匹配数，两者不一定相等（`valid_matches` 已在 `geometricVerification` 里过滤过一次）。内点率因此失真，`max_icp_inlier_ratio`（`geometric_verifier.hpp:702-704` 的退化判据）也就失去意义。

**修法**：分母改为 `dummy_matches.size()`。

---

### L7（P1）ICP 用 `dummy_matches`，没有描述子验证

`loop_detector.cpp:524-527`：

```cpp
std::vector<cv::DMatch> dummy_matches;
for (int i = 0; i < pts3d_query_cam_valid.size(); ++i)
    dummy_matches.emplace_back(i, i, 0);      // 索引一一对应
```

这意味着"当前帧第 i 个深度点"与"匹配帧第 i 个深度点"被当成同一个物理点。两帧关键点集合不同、顺序不同、数量不同时，这个对应关系是随机成立的。`:501-513` 的注释自承这是"导致错误旋转（38.6°）"的原因，但采用的补救是"跳过 ICP 直接降级到 EM"，而不是修好 ICP。

**修法**：ICP 的输入必须是**经过描述子验证的对应关系**。已经有现成的 `matches`（`matchDescriptors()` 双向交叉验证 + Lowe 比率的产物），直接用 `matches` 而不是 `dummy_matches`。若担心数量不足，用 `valid_matches`（`geometricVerification` 里已按 z>0 过滤的那份）。

---

### L8（P1）SIGSEGV 捕获 + 120 行注释块

`pose_graph_optimizer.cpp`：

- `:1196-1315` —— 120 行 `/* ... */` 整块注释掉的 `[PGO CRASH-DIAG]` 诊断代码
- `:16-24` 文件作用域 `static sigjmp_buf pgo_sigsegv_jump_buf;` + `static void pgo_sigsegv_handler(int)`
- `:1325-1421` `#ifndef _WIN32` 段：`sigsetjmp` 保存跳转点 → 安装 SIGSEGV/SIGBUS/SIGABRT 处理器 → `optimize()` → 崩溃时 `siglongjmp` 回来 → 清理边缓存 → 返回 0

在优化器内部捕获段错误然后继续运行，会掩盖真正的内存错误（悬空指针、堆损坏），并把"程序已经坏了"表现为"这次 PGO 没收敛"。这类处理放在库代码里是不可接受的。

**修法**：
1. 删除 `:1196-1315` 的注释块（git 里有历史，不需要留在源文件里）。
2. 删除 `sigsetjmp` / `sigaction` 保护，崩溃就让它崩 —— 崩溃日志比"静默返回 0"有用得多。
3. 保留 `:1072-1142` 与 `:1144-1180` 的顶点/边校验（那是正向的数据校验，不是吞异常），以及 `:1281-1314` 的 dangling edge 预检查（同样有价值，但需要从注释块里拿出来、重新启用）。

---

### L9（P2）死代码 / 未接线

| 符号 | 位置 | 状态 |
|---|---|---|
| `correctLoopWithScale()` | `loop_detector.cpp:908-957` | 全仓无调用点。它做的 `scale = actual_dist / estimated_dist` 正是 EM 路径缺的那一步 |
| `computeCandidateConsistency()` | `loop_detector.cpp:1113+` | 全仓无调用点，`require_candidate_agreement` / `consistency_warning_threshold` 因此是 dead key |
| `Tracking::initLoopDetection()` 的 `pgo_cfg` | `tracking.cpp:3260` | 只在构造时用一次，之后无法改 |
| `LoopDetectorConfig::verbose` | `loop_detector.hpp` | 已接线，但 `geometric_verifier.hpp` **45 处 `std::cerr` 全部无守卫**（见 L10） |

---

### L10（P1）`geometric_verifier.hpp` 45 处 `std::cerr`，零守卫

```
grep -c "std::cerr"          src/.../geometric_verifier.hpp  →  45
grep -c "if (config_.verbose)" src/.../geometric_verifier.hpp  →   0
grep -n "bool verbose"        src/.../geometric_verifier.hpp  →  （无命中）
```

`GeometricVerifierConfig`（`geometric_verifier.hpp:27-52`）里**没有 `verbose` 字段**，所以这个类从设计上就没有开关。`LoopDetector` 侧的 59 处守卫（B2 阶段已加）全部失效在这一点上——真正的刷屏发生在被它持有的 `GeometricVerifier` 内部。

量级：一次回环检测最多 5 个候选 × (Essential RANSAC 200 轮 + ICP 全对 + SVD 精化 150 轮)，每轮若干行；73 个关键帧累计下来 stderr 体量是十万行级。这既是 `1 s/帧` 的主要来源之一，也让真正的失败信息被淹没（R-017 的原始动机）。

**修法**：给 `GeometricVerifierConfig` 加 `bool verbose = false;`，用 `scripts/guard_diag_logs.js --guard=config_.verbose` 施加守卫，保留错误/失败日志无条件输出（R-021）。

---

### L11（P2）`selectBestFromVerified()` 排序用非严格弱序

```cpp
// loop_detector.cpp:1062-1070
std::sort(sorted.begin(), sorted.end(),
          [](const VerifiedCandidate& a, const VerifiedCandidate& b) {
              double score_gap = std::abs(a.quality_score - b.quality_score);
              double larger_score = std::max(a.quality_score, b.quality_score);
              if (larger_score > 0 && score_gap / larger_score > 0.05) {
                  return a.quality_score > b.quality_score;
              }
              return a.is_icp && !b.is_icp;
          });
```

这个比较器不满足严格弱序：若 `a.score=100, b.score=98`（差 2%，走第二条分支，`a.is_icp=false, b.is_icp=true` → `comp(a,b)=false`），而 `b` 与 `c.score=99`（差 1%，同样走第二条，`b.is_icp=true, c.is_icp=false` → `comp(b,c)=false`），同时 `a` 与 `c` 差 1% → `comp(a,c)=false`。此时可能构造出 `a<b<c` 且 `c<a` 的循环。`std::sort` 对非严格弱序的比较器是**未定义行为**（常见表现是越界读写）。

**修法**：先按 `is_icp` 分组（`true` 优先），组内按 `quality_score` 降序，用两个明确的比较器串联：

```cpp
if (a.is_icp != b.is_icp) return a.is_icp;      // ICP 优先
return a.quality_score > b.quality_score;        // 其余按分数
```

若确实需要"分数接近时让 is_icp 决定"，用 `stable_sort` + 明确的两级 key（`std::make_pair(is_icp, quantized_score)`），不要写带阈值的分支。

---

## 四、修复顺序（R1 → R7，顺序不可调）

> 原则：先把"有尺度的路径"接通并修对，再谈"用不用它"。在 L1/L2 修好之前调任何阈值都是在假尺度上调参。

| 步骤 | 内容 | 对应 | 验收判据 |
|---|---|---|---|
| **R1** | 修 L1 双重变换 + L6 分母 | L1、L6 | 单测：给定已知 `T_12` 与两帧相机系点，`geometricVerificationWithSim3` 解出的 R/t 与真值误差 < 1% |
| **R2** | 修 L2：Sim3 输入改用 `keyframe_db_` 自身观测（方案 A） | L2 | 日志出现 `[GeoVerif Sim3]` 分支的执行记录；`out_is_icp=true` 的比例 > 0 |
| **R3** | 修 L3：删 odom 边 `setLevel(1)`，`initializeOptimization` 显式 0 | L3 | `[PGO] optimize: ... loop_edge_data_=N` 后 `chi2` **不再恒 0**；"最大修正"的量级与漂移量匹配 |
| **R4** | 修 L4：`addLoopEdge` 返回 `bool` | L4 | `mnDetectedLoops_` 与 `loop_edge_data_` 同步增长 |
| **R5** | 修 L5：`GeometricVerifierConfig` 并入 `LoopDetectorConfig` | L5 | 改 `config/loop_detector.yaml` 里 `min_icp_inlier_ratio` 能改变行为（打印实测阈值确认） |
| **R6** | 修 L10：加 `verbose` 字段 + 45 处守卫 | L10 | 关掉 verbose 后回环检测的 stderr 行数 < 5；错误日志仍在 |
| **R7** | 修 L7 + L8 + L9 + L11 | L7、L8、L9、L11 | L7：ICP 用真实 `matches`，同一对候选的 ICP 与 Essential 判定不再矛盾；L8：源文件从 1497 行降到 ~1350，无信号捕获；L9：删除或接线死代码；L11：比较器满足严格弱序（可加断言验证） |

**注意**：R3 落地后，如果 `chi2` 从 0 变成很大的值（例如 1e6），说明回环边本身是错的（L1/L7 的结果），此时**不要**去调 `robust_kernel_delta` 掩盖它 —— 那会把"错误约束"变成"被 Huber 压小的错误约束"，一样是错的。正确做法是拒绝该回环边。

---

## 五、验收标准

前置：前端 R1–R5 已修（否则本层的所有指标都不可解释）。

1. **规模指标**（fr1_desk2，640 帧，`--slam`）
   - 路径总长 8–14 m（真值约 5–12 m）
   - ATE rmse ≤ 0.10 m、RPE_trans rmse ≤ 0.03 m（`scripts/evaluate_trajectory.py --align`，RGB-D **不估尺度**，R-024）
   - 起终点漂移 ≤ 0.15 m
   - 单帧耗时 ≤ 150 ms

2. **回环层自身指标**
   - `loop_edge_data_` 与实际有效回环次数一致（不是"调用次数"）
   - 每次 PGO 的 `chi2` 首轮 > 0 且逐轮下降（不是恒 0）
   - "最大修正"与累计漂移同量级（不是 0.08 m 对 4 m 漂移）
   - 同一对候选的 ICP 与 Essential 判定方向一致（不再出现 6.6% vs 91.3%）

3. **不退化**
   - 关闭 `--slam`（纯 VO）的指标不得比修复前更差
   - 不出现新的 SIGSEGV / 未捕获异常

---

## 六、风险与回滚

- 项目**不是 git 仓库**，回滚只能靠副本。开工前：
  ```
  cp src/sfc_core/src/loopdetection/loop_detector.cpp /tmp/loop_detector.pre-LR.cpp
  cp src/sfc_core/src/loopdetection/pose_graph_optimizer.cpp /tmp/pgo.pre-LR.cpp
  cp src/sfc_core/src/frontend/tracking.cpp /tmp/tracking.pre-LR.cpp
  cp config/loop_detector.yaml /tmp/loop_detector.pre-LR.yaml
  ```
- **R1 会让 Sim3 从"永不执行"变成"执行"**，行为变化最大。建议 R1+R2 一起验证，单独启用 R1 时 Sim3 仍不会被调用（因为 L2 未修），无法观察效果。
- **R3 会让 PGO 真正开始移动顶点**。若此时回环边是错的（L1/L7 未修好），轨迹会**变差**——这是预期的，说明约束本身有问题，不能靠调 level 掩盖。
- `pose_graph_optimizer.cpp` 的 SIGSEGV 防护删除后，如果 g2o 真的有内存问题会直接崩。建议删除前先在有工具链的机器上跑一次 ASAN 构建，确认无悬空指针。
- **不要同时改 L1 和 L7**：两者都改变"回环边的数值"，同时改无法判断哪个对。

---

## 七、与 CODE-RULES 的关系

| 规则 | 本层违规点 |
|---|---|
| **R-007**（单文件 ≤ 400 行） | `loop_detector.cpp` 1321、`pose_graph_optimizer.cpp` 1497、`geometric_verifier.hpp` 1103、`place_recognition.hpp` 973 —— 全部超标 |
| **R-017 / R-021**（诊断输出受开关控制，错误日志不受） | `geometric_verifier.hpp` **45 处 `std::cerr`、0 处守卫**（`GeometricVerifierConfig` 里根本没有 `verbose` 字段）；`pose_graph_optimizer.cpp:931-932` 裸打印 |
| **R-018**（魔法数字具名常量） | `kEMFallbackTranslationScale = 1.0`（`:631`）、`MAX_SPATIAL_DIST_FOR_LOOP = 8.0`（`:750`）、`frames_between < 30`（`:766`）、`queryWithConsistency(..., 3, 10, 1.0)`（`:742`）、`min(5ul, dbow_results.size())`（`:751`） |
| **R-019 / R-022 / R-023**（配置单一来源、无 dead key、单一归属文件） | `config/geometric_verifier.yaml` 整份无消费者（L5）；`require_candidate_agreement` / `consistency_warning_threshold` 无消费者（L9） |
| **R-025**（进图优化的约束必须有真实尺度证据） | L7 的 `dummy_matches`、L6 的 `kEMFallbackTranslationScale = 1.0` 都是"拿不到可靠基线就用常数顶替"的反例 |
| **R-027**（同一度量只有一个事实来源） | `loop_edge_data_`（C++ 容器）与 g2o 图内边数在去重后不一致，且没有对齐机制 |

建议新增 **R-028**（第 8 节，MUST）：

> 图优化中"哪些约束参与求解"必须由 `information()` / 鲁棒核 / 显式拒绝来表达，**不得用 `setLevel()` 把整类约束移出优化**。使用 `initializeOptimization()` 时必须显式传入 level，并在同一处注释说明该 level 的语义。任何"某类边被批量设为 level>0"的写法必须附带一条测试，证明剩余约束仍足以确定全部顶点。

理由：`pose_graph_optimizer.cpp:272` 把 63 条里程计边全部设为 level 1，使 `initializeOptimization()`（默认 level 0）只激活 1 条回环边，优化退化为"解一条边"，`chi2` 恒 0 却计为成功——这是本项目第三次因"配置/参数写了却不生效"导致的静默失效（前两次是 `octree_orb.yaml` 不存在、`TrackingConfig::enable_local_ba` 无消费者）。

---

## 八、遗留

1. 本机（Windows）**无任何 C++ 工具链**，本文件所有结论来自静态阅读 + 用户提供的运行日志，**一行代码都没有编译过**。
2. `docs/` 下现共四份计划：
   - `PLAN-tracking-drift-fix.md` —— 纯 VO 阶段记录（阶段 A/B1/B3/C1 已落地）
   - `PLAN-slam-loop-pgo-fix.md` —— 首次接回环时的诊断（未实施）
   - `PLAN-frontend-pose-fix.md` —— 前端 R1/R2/R3/R4/R5（379 行，未实施）
   - `PLAN-loop-detection-fix.md` —— 本文件（L1–L11 根因 + R1–R7 修复步骤）
3. **前端未修好之前，本文件的 R1–R7 全部实施也无法让 fr1_desk2 收敛**。建议的执行顺序是：前端 E1（R5）→ E2（R1）→ E3（R2）→ 本文件 R1–R4。
4. 未验证但有风险的点（需要在有工具链的机器上确认）：
   - `selectBestFromVerified()` 的排序 lambda（`loop_detector.cpp:1062-1070`）混用"分数差 >5% 时按分数、否则按 is_icp"，不是可证明的严格弱序，`std::sort` 对非严格弱序是未定义行为。已升为 **L11**（见第一章表）
   - `SimpleKDTree2D`（`:76-139`）实际是 O(N²) 暴力最近邻，每次几何验证都对两帧各建一次
   - `Sim3Solver` 的误差阈值写死 `20.0f`（`sim3_solver.cpp:59-63`），不按 octave 缩放


---

## 九、执行记录（2026-11-10）

**范围**：R1(L1+L6) → R2(L2) → R3(L3) → R4(L4) → R5(L5) → R6(L10) → R7(L7+L8+L9+L11)。
**验证状态**：Windows 本机**无任何 C++ 工具链**，全部改动未编译；只做括号配对与遗留符号静态核对
（`loop_detector.cpp` / `loop_detector.hpp` / `pose_graph_optimizer.cpp` / `pose_graph_optimizer.hpp` /
`geometric_verifier.hpp` 均 `final brace=0 paren=0 brack=0`）。

### R1 · L1 + L6

- **L1**：`geometricVerification()` 构造 Sim3 输入时不再做第二次 world→camera。原来调用侧已经
  用 `R_cw * pt_world + t_cw` 转过一次，被调函数 `geometricVerificationWithSim3()` 内部又各转一次
  （形参名就叫 `*_world`），旋转平移施加两遍把点云搬到相机背后 —— 这是 covis 路径"永远失败、
  只能回落 ICP+EM"的根因。
- **L6+L7**：`valid_matches` 改用 `matches[i].queryIdx/trainIdx` 取点（原来误用循环下标），
  并按传给 ICP 的压缩序号重映射 `queryIdx/trainIdx`；ICP 内点率分母由 `matches.size()` 改成
  `valid_matches.size()`。

### R2 · L2

`geometricVerification()` 的阶段 2（描述子 matches + `keyframe_db_` 自带的世界坐标点）
与 Sim3 块整体移出 `if (covis_mgr != nullptr)` 守卫，只有阶段 1（confirmed map point + KD-tree）
仍需要 `covis_mgr`。`Tracking::detectAndCorrectLoop()` 继续传 `nullptr`，但现在是有意为之 ——
原实现把阶段 2 也写在守卫里，于是这条路径上 `covis_matches` 恒空、Sim3 永不进入。

### R3 · L3

- `addOdometryEdge()` 里的 `edge->setLevel(1)` 删除。它配合 `optimize()` 不带参数的
  `initializeOptimization()`（默认 level 0），效果是**所有里程计边被整类移出优化**，
  图上只剩 1 条回环边 —— 一条边的图 chi2 必然恒 0。
- `initializeOptimization()` 改为显式 `initializeOptimization(0)`。

### R4 · L4

`addLoopEdge()` 由 `void` 改为 `bool`：10 处拒绝/去重点全部 `return false`，成功路径
`return true`。`tracking.cpp` 与 `covis_manager.cpp` 两处调用点都在 false 时直接返回，
不再"检测到回环但边没进图"却计一次回环。

### R5 · L5

几何验证阈值回配置：`config/geometric_verifier.yaml` 一直无人读取，阈值写死在
`loop_detector.cpp` 的构造函数里。现在 `LoopDetectorConfig` **直接持有**
`GeometricVerifierConfig` 成员，由 `loadAllConfigs()` 注入，两条回环路径自动同步。
`min_inliers` 2→12、`min_inlier_ratio` 0.15→0.25、`min_icp_inliers` 2→12。
同一份 yaml 的 `min_depth` 0.05→0.1、`max_depth` 30→12，与 Tracking 保持一致
（此前 yaml 没有消费者，这个 P0-1 级的不一致一直没被发现；接上之后它就成了生效值）。

### R6 · L10

`geometric_verifier.hpp` 的 46 处 `std::cerr` 中 44 处加 `if (config_.verbose)` 守卫；
2 处跳过：一处是注释里提到的说明文字，一处是"内点数不足"（按 R-011 保持无条件）。

### R7 · L7 + L8 + L9 + L11

- **L7**：`icp_inliers == 0` 时的"hybrid fallback"（`T_hybrid = R_icp + frame_dist * t_icp_dir`）
  整段删除，改为降级到 Essential Matrix 并打一条 verbose 日志。原代码在注释里写着"已删除
  hybrid fallback"，但代码还在。
- **L8**：`pose_graph_optimizer.cpp` 的 `sigjmp_buf` / `sigaction` / SIGSEGV 捕获整块删除，
  只保留 C++ `try/catch`。理由：段错误发生时 g2o 内部已处于半更新状态，`longjmp` 跳过栈展开
  后返回的是损坏的图，后续每帧都在损坏状态上"优化" —— 崩溃被换成静默的错误轨迹，比直接崩更难查。
  同时把原本被注释掉的 dangling-edge 预检查**复活成真代码**（必须早于 `initializeOptimization()`，
  否则 `_activeEdges` 里留悬空指针）。文件 1501 → 1323 行。
- **L9**：`computeCandidateConsistency()` 接线进 `selectBestFromVerified()`（`require_candidate_agreement` /
  `consistency_warning_threshold` / `max_spatial_dist_for_consistency` 三个键首次真正生效）；
  `correctLoopWithScale()` 与 `scale_min` / `scale_max` 删除 —— 它的尺度逻辑已被 EM 路径的
  深度实测基线取代，保留会形成第二套尺度真值。
- **L9 附带（原 D2-b）**：EM 降级路径写死的 `kEMFallbackTranslationScale = 1.0` 删除。
  现在用每对有效对应点算出的 `t = p_train - R_em * p_query` 投影到单位方向取中位数作为基线；
  样本不足 3 对或基线非有限时**直接拒绝这次 EM 回环**，不再编一个数顶上。
- **L11**：排序比较器改为严格弱序（`is_icp` → `quality_score` → `kf_id` 三级 key），
  原实现 `return a.is_icp && !b.is_icp;` 在相等时返回 true，`std::sort` 属未定义行为。

### 未完成 / 需在有工具链的机器上确认

1. **编译与实测**：一行都没编译过。L8 删除 SIGSEGV 防护后建议先在有工具链的机器上跑一次 ASAN
   再上真实序列。
2. **不要同时判定 L1 与 L7 的贡献**：两者都改变回环边数值，本文件 §八 已警告不能同时改；
   实际执行时两者都改了，若结果异常需要按本记录的提交顺序分别回退定位。
3. `SimpleKDTree2D` 实际是 O(N²) 暴力最近邻、`Sim3Solver` 的 `20.0f` 误差阈值不按 octave 缩放 ——
   两项风险本文件 §八 已列出，本次未处理。
4. `config/covis_manager.yaml` 的 `loop_detection_interval: 9999` 仍在。这意味着 **CovisManager
   那条回环路径实际上从不检测回环**（注释写的是"纯 VO 模式"）。它与
   `PLAN-tracking-drift-fix.md` 的 B2 是同一个待决策问题（纯 VO 还是完整 SLAM），本次不动。
5. 本文件 §九 建议新增的 **R-028** 已采纳为代码注释原则（见 `addOdometryEdge()` 内的说明），
   但尚未写入 `CODE-RULES.md`。
