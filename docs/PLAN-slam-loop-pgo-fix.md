# 完整 SLAM（回环 + PGO）退化与性能问题：诊断与修复计划

> 前一份计划 `docs/PLAN-tracking-drift-fix.md` 覆盖的是**纯 VO** 阶段（路径总长 478.6 m）。
> 本文覆盖**开启 `--slam` 之后**的新问题：路径总长恶化到 1165.1 m、单帧耗时约 1 s。
> 状态：只做诊断，尚未改动任何代码。

---

## 一、结论摘要

开启完整 SLAM 后，轨迹**比纯 VO 更差**，并且慢了约 20 倍。这不是"回环参数没调好"，而是**回环给出的相对位姿在物理上没有意义，却被当作 6-DoF 约束写进了位姿图**；同时**位姿图本身只有一个回环边**，优化解不出任何修正；而每次"回环"都仍会触发一次全量 g2o 重建 + 全地图点搬移。

一句话：**回环检测判定"通过"了，但算出来的位姿是错的；PGO 没解出东西，但写回地图的副作用照常发生。**

四条主干原因：

| # | 问题 | 后果 | 位置 |
|---|---|---|---|
| 1 | Essential 恢复的平移只有方向、被固定成 1.0 m | 回环边把相机"搬"了 1 m，无论真实相隔多远 | `loop_detector.cpp:622-639`、`geometric_verifier.hpp:225-237` |
| 2 | 几何验证阈值全部写死在 `loop_detector.cpp`，`geometric_verifier.yaml` 整份无人读取 | `min_inlier_ratio=0.15`、`icp_reproj_thresh_px=35` 让 15% 的垃圾匹配擦线通过 | `loop_detector.cpp:155-172` |
| 3 | `addLoopEdge` 去重后 `return`，调用方无法区分"加边"与"跳过" | 43 次"回环"里 `loop_edge_data_` 恒为 1，其余 42 次是空转 | `pose_graph_optimizer.cpp:769-831`、`tracking.cpp:3365-3395` |
| 4 | PGO 位姿写回地图点时按"第一个观测者"选一份 delta | 多观测点与其它观测帧错位，地图里长出重复结构 | `tracking.cpp:3464-3480` |
| 5 | 地图点容器只增不删（13082 个点，绝大多数是单次观测） | 耗时随帧数上升；同一处表面存在多份不关联的点 | `tracking.cpp:943-1009`、`:826-857`、`:3218-3231` |

---

## 二、实测数据

`fr1_desk2`，640 帧，同一份代码、同一份数据，只差 `--slam`：

| 指标 | 纯 VO | 完整 SLAM | 真值参考 | 判断 |
|---|---|---|---|---|
| 路径总长 | 478.6 m | **1165.1 m** | 约 5–12 m | 恶化 2.44 倍，超标约 100–230 倍 |
| 起终点漂移 | 4.300 m | 1.341 m | — | 反而改善（无意义，见下） |
| 关键帧数 | 32 | 73 | — | 关键帧变密 |
| 平均关键帧间隔 | 1.7957 m | 1.9577 m | 0.15 m（配置值） | 超标 13 倍 |
| 最大关键帧间隔 | — | 2.7277 m | — | — |
| 累计"回环" | — | 43 次 | — | 其中 42 次无新约束 |
| 单帧耗时 | 约 50 ms | **约 1 s** | — | 慢约 20 倍 |

"起终点漂移改善"是假象：路径在两个位姿之间来回折返，起点与终点恰好都落在其中一个簇附近。**路径总长才是这里唯一诚实的指标**——而按 R-024，验收必须改用 ATE/RPE。

---

## 三、根因逐条

### 原因 1：回环给的平移恒为 1.0 m（最致命）

**现象**：日志里每一次 `[GeoVerifier] Relative pose: |t|=1.0 m`，`|t|` 永远是 1.0。

**机制**：回环几何验证最终走的是 Essential 矩阵（2D-2D）分支。

- `src/sfc_core/include/sfc_core/loopdetection/geometric_verifier.hpp:161-163` `cv::findEssentialMat(...)` 得到 E；
- `:225` `cv::recoverPose(E_, inlier_pts1, inlier_pts2, K, R, t)` —— **Essential 矩阵只约束到尺度，恢复出的 `t` 天然是单位向量**；
- `:228-237` 直接 `block<3,1>(0,3) = t_eigen`，**没有乘任何基线**；
- `:239-267` 的 SVD 退化分支更明确：`t_eigen /= t_norm`，日志写"已归一化，仅方向可靠"。

于是这个位姿被原样交给 PGO 当 6-DoF 约束用。

**更糟的是 EM 回退路径还把它固定成 1.0 m**：`src/sfc_core/src/loopdetection/loop_detector.cpp:622-639`

```cpp
constexpr double kEMFallbackTranslationScale = 1.0;
Eigen::Vector3d delta_t = pose_curr.t - pose_matched.t;
double frame_dist = delta_t.norm();
...
T_em.block<3,1>(0,3) = kEMFallbackTranslationScale * t_em_dir;
```

注释写的是"不再使用漂移的 odometry 估计"，但结果是：**无论两个关键帧真实相隔 0.2 m 还是 3 m，回环边都声称它们相隔 1 m**。位姿图被灌入一条错误的绝对尺度约束，优化只能把整段轨迹拉向这个错误值。

而紧接着的帧间距离闸门（`:644-654`）用的又正是这个被否决的 `frame_dist`（来自漂移位姿）——判据自相矛盾。

**修法**：RGB-D 有真实深度，回环相对位姿**不应该**走 2D-2D Essential 定尺度。应优先走 3D-3D（ICP / Sim3），并用匹配地图点的中位深度给出真实基线：

```
baseline = median( ||p_world_query - p_world_train|| )  // 来自实测深度
T_em.translation = baseline * t_em_dir
```

若拿不到可靠基线，**这条回环必须被拒绝**，而不是回退到 1.0 m。

---

### 原因 2：几何验证阈值写死，`geometric_verifier.yaml` 是死配置

**现象**：日志里 `Essential inliers=86/549 (15.7%) PASS`、`85/575 (14.8%) FAIL: ratio=0.15 < min_ratio=0.15` —— 15.x% 的匹配刚好擦着线过。

**机制**：`src/sfc_core/src/loopdetection/loop_detector.cpp:155-172` 构造 `GeometricVerifierConfig` 时只从 `LoopDetectorConfig` 抄了 5 个字段，其余全是字面量：

```cpp
GeometricVerifierConfig geo_config;
geo_config.ransac_iterations = config_.ransac_iterations;
geo_config.threshold_px      = config_.ransac_threshold_px;
geo_config.min_inlier_ratio  = 0.15;   // 写死
geo_config.min_inliers       = 2;      // 写死
geo_config.min_icp_inliers   = 2;      // 写死
geo_config.min_icp_inlier_ratio = 0.10;// 写死
geo_config.icp_reproj_thresh_px = 35.0;// 写死（yaml 是 25.0）
geo_config.max_icp_inlier_ratio = config_.max_icp_inlier_ratio;
geo_config.min_depth = 0.1;            // 写死
geo_config.max_depth = 12.0;           // 写死
```

`config/geometric_verifier.yaml`（`:44 min_inliers: 2`、`:54 min_inlier_ratio: 0.15`、`:83 icp_reproj_thresh_px: 25.0` …）被 `loadGeometricVerifierConfig` 正常读进 `AllConfigs::geometric_verifier`，但**全仓没有任何一处把它传给 LoopDetector**——只有 `config_loader.cpp:624` 一行赋值，之后再无消费者。

三重违规叠加：R-019（两处默认值）、R-022（yaml 键无消费者）、R-023（配置有归属文件但没接线）。

**附带**：`icp_reproj_thresh_px` 从 yaml 的 25.0 放宽到 35.0，直接把像素误差 25–35 px 的错配算成内点 —— 这就是日志里 `ICP inliers=62/946 (6.6%)` 与 `Essential inliers=1108/1214 (91.3%)` 同时出现时，ICP 反而先失败的原因：它的判据被稀释了，但仍然过不了 `min_icp_inlier_ratio`。

**修法**：

1. `LoopDetector` 构造函数接受一个完整的 `GeometricVerifierConfig`（或 `AllConfigs::geometric_verifier`），不再自己拼；
2. `LoopDetectorConfig` 里删掉与几何验证重复的字段，避免两个来源；
3. `min_inlier_ratio` 不应低于 0.25——15% 内点率意味着 85% 是误匹配，几何上不构成回环。

---

### 原因 3：回环边被去重后仍被当成成功

**现象**：

```
[PGO] addLoopEdge 跳过（loop_edge_ids_中已存在）: 623 -> 621
[Loop] 回环修正完成：KF=623 ↔ 621 iterations=16 修正关键帧=65 累计回环=35
```

以及贯穿全程的 `[PGO] optimize: vertices=65 edges=64 odom_edge_data_=63 loop_edge_data_=1` —— 43 次回环，位姿图里从头到尾只有 **1 条**回环边。

**机制**：

- `src/sfc_core/src/loopdetection/pose_graph_optimizer.cpp:769-797`：命中 `loop_edge_key_set_` 就打印一句然后 `return;`。`addLoopEdge` 返回类型是 `void`，**调用方无从得知这次到底加没加**。
- `:798-831` 旧 O(n) 兜底同样 `return;`。
- `src/sfc_core/src/frontend/tracking.cpp:3365-3395` `detectAndCorrectLoop()` 调完 `addLoopEdge` **无条件**继续 `optimize()` → `applyOptimizedPosesToFrontend()` → `mnDetectedLoops_++` → 打印"回环修正完成"。

于是 42 次空转全部走了完整流程：全量重建 g2o 图、再遍历 73 个关键帧 + 13082 个地图点搬一次。

**同样致命的是另一半**：`optimize()` 的每一轮 `chi2` 都是 `0.0`，`lambda` 一路爬到 `2485.7 / 603298.6 / 70619.3`。chi2 恒为 0 说明**约束之间没有任何张力**——1 条回环边 + 63 条里程计边，而里程计边本身来自同一份漂移轨迹，自洽得不能再自洽。PGO 没有可优化的东西，所以"最大修正"只有 0.08 m / 0.02 m / 0.003 m。

**修法**：

1. `addLoopEdge` 改为返回 `bool`（或返回枚举 `ADDED / DUPLICATE / REJECTED`），调用方据此决定是否优化；
2. `detectAndCorrectLoop()` 在 DUPLICATE 时直接返回，不做优化、不做写回、不计数；
3. 回环边数量与 chi2 都为零时应记一条显式警告，而不是打印"修正完成"。

---

### 原因 4：PGO 位姿写回地图点的策略错误（B2 引入）

**现象**：`[  631/640] pose=(1.120, -0.511, -0.528) inliers= 821 kf=Y` 与 `[  633/640] pose=(-0.986, 0.356, 0.838) inliers= 904 kf=Y` 反复交替，相邻帧差 `|t|=2.65899 m`。67 ms 内移动 2.65 m = 40 m/s，物理上不可能。

关键在于**两个位姿都能在同一张地图里找到 800–1000 个内点**。如果地图只有一份结构，只有一个位姿能对上；两处都能对上，说明**地图里同一片场景存在两份重叠结构**。

**机制**：`src/sfc_core/src/frontend/tracking.cpp:3464-3480`

```cpp
// 地图点只搬一次，delta 取自"第一个观测者"
const int owner_kf = mp.observations.front().first;
const Eigen::Matrix4d& d = delta_by_kf[owner_kf];
mp.world_point = (d * p.homogeneous()).head<3>();
```

一个地图点通常被多个关键帧共同观测。按"第一个观测者"的 delta 搬，这个点在**其它观测帧的坐标系里就必然错位**。43 轮回环、每轮 65 个关键帧各带一份 delta 累积下来，同一片场景被搬成多份互相错开的结构——正好解释"两个位姿都能找到上千内点"。

**另外两个缺陷**（同一函数）：

- `:3408-3425` 只有一道"整次丢弃"的安全阀：`max_shift > 10.0 m` 才 `return -1`。10 m 以内的任何畸形修正都放行，没有单帧上限、没有连续性检查。
- `:3432-3457` 关键帧改写是**逐条立即生效**的：一旦中途某帧修正量离谱，前面的已经写进 `mAllKeyFrames_` / `mvpLocalKeyFrames_` / `mpMap_->getKeyFrame()->setPose` / `loop_detector_->updateKeyFramePose`，没有事务性、没有回滚。

**修法**（两条路线，建议先做保守的 A）：

- **A（保守，推荐先落）**：回环只修正**关键帧位姿**，**不搬地图点**。地图点保留原世界坐标，`updateWorldPoints()` 在下一帧用新位姿重算。代价是短期内重投影误差变大，但不会长出重复结构。等 ATE 稳定后再考虑搬点。
- **B（完整）**：搬点用**世界系下的加权平均**而不是取某一份 delta：
  ```
  p_new = Σ_k w_k * (T_k_new * T_k_old^{-1} * p) / Σ_k w_k
  ```
  `w_k` 取该点在第 k 帧的观测深度倒数（近的权重大）。同时把写回改成"先全量算、再统一提交"，任何一帧超限就整次放弃。

---

### 原因 5：B3 跳变守卫与 PGO 写回的时序错位

`src/sfc_core/src/frontend/tracking.cpp` 的执行顺序：

```
:691-711  poseOptimization()
:713-736  B3 跳变守卫（FRAME_JUMP_MAX_TRANSLATION_M = 0.5）
:738-747  tracking_ok 判定
:762      needNewKeyFrame()
:778-779  createNewKeyFrame()
:793      detectAndCorrectLoop()   ← PGO 在这里改写 mCurrentFrame_->pose
:799-801  mLastFrame_ = mCurrentFrame_
```

守卫跑在 PGO 之前，**PGO 修改后的位姿永远不会再过一次检查**。这就是 2.65 m 跳变能进轨迹的直接原因。

**修法**：`detectAndCorrectLoop()` 之后补一道同样的跳变检查；超限则整次回环回滚（连同 `mnDetectedLoops_` 计数）。

顺带一个可疑点：日志里 `[  623/640] inliers=   8` 只有 8 个内点仍判 `kf=Y`、`tracking_ok=true`。B3 守卫对 `track_method == "RELOC"` 显式豁免——重定位确实该允许跳，但**豁免范围过大**，需要单独核实 RELOC 之后的运动模型是否被正确重置。

---

### 原因 6：性能——1 s/帧的四个来源

按影响从大到小：

1. **`geometric_verifier.hpp` 全文件没有 verbose 守卫**（`:189/:194/:237/:241/:266/:271/:929/:934` 等）。一次回环检测刷几十行，640 帧的 stderr 体量约 18 万 token。R-017/R-021 违规。
2. **每次"回环"都全量重建 g2o 图**：`addLoopEdge` 置 `optimizerNeedsRebuild_ = true`（`pose_graph_optimizer.cpp:947`），`optimize()` 开头无条件 `rebuildOptimizer()`（`:972-974`）。65 顶点 + 64 边的 `BlockSolver` / `LinearSolver` 全部重新构造。43 次就是 43 次全量重建——其中 42 次本该什么都不做（原因 3）。
3. **每次写回遍历全地图**：73 个关键帧 × （`mAllKeyFrames_` 哈希 + `mvpLocalKeyFrames_` deque 线性扫描 + `mpMap_->getKeyFrame()` 查找 + `loop_detector_->updateKeyFramePose()`）+ 13082 个地图点各一次 4×4 变换与两处哈希写入。
4. **每次检测 5 个候选**，每个候选一遍 Essential RANSAC（200 次迭代、约 1200 匹配）+ ICP（约 1000 对 3D-3D）+ 失败后再跑一次 SVD 回退。`loop_detector.cpp:751` 的 `std::min(5ul, dbow_results.size())` 还把 `candidate_pool_size` 又绕开了一次。

**预期改善**：修完原因 3 之后，73 次检测里约 72 次会在去重处直接返回，重建与写回次数从 43 降到个位数——这一项就能拿回大部分时间。日志守卫再消掉 stderr 开销。

---

### 原因 7：地图点容器只增不删，每帧深度点都变成永久地图点

这一条不直接让轨迹变差，但它解释了**为什么耗时随帧数线性上升**，以及**为什么"同一片场景能对上两个位姿"更容易发生**。

- `src/sfc_core/src/frontend/tracking.cpp:943-1009` `updateLastFrame()`：对上一帧**每一个有效深度且无关联地图点**的特征点调用 `createMapPoint(..., obs_count=1)`，这些点直接进 `mAllMapPoints_`。
- `tracking.cpp:826-857` 帧末又对当前帧做同一件事，再次批量写入 `mAllMapPoints_`。
- `tracking.cpp:3218-3231` `cleanOldMapPoints()` 只做一件事：把 `observed_count == 0` 或 `== 1` 的点标成 `is_bad`，**不删除**。
- 全仓 grep `mAllMapPoints_.erase` **零命中** —— 这个容器从进程启动到结束只增不减。

于是 `mAllMapPoints_` 在 fr1_desk2 上累积到 **13082 个点**，其中绝大多数是"某帧深度测了一次、之后再没被观测"的单次点。ORB-SLAM3 里这类点是 `mlpTemporalPoints`（帧内临时，用完即弃），本项目把它们写成了永久地图点。

后果有两层：

1. **性能**：`applyOptimizedPosesToFrontend()` 每轮回环要对全部 13082 个点做一次 4×4 变换 + 两次哈希写入；`searchLocalPoints()` 在候选不足时还会做全局扫描。
2. **正确性**：`is_bad` 的点不会被 `cleanOldMapPoints()` 复活，但**新帧又会用新的 `mp_id` 重新建一遍同位置的点**——同一处物理表面在地图里存在多份互不关联的点。这正是"两个位姿都能找到 800–1000 内点"的第二个来源（第一个是原因 4 的错位搬点）。

**修法**（D6 阶段）：

1. `updateLastFrame()` 与帧末重建的临时点**不应进 `mAllMapPoints_`**，应放进独立的 `mlpTemporalPoints` 池，帧结束即丢弃（对齐 ORB-SLAM3 语义）；
2. `cleanOldMapPoints()` 改名 `retireBadMapPoints()`，真正 `erase` 掉 `is_bad && observed_count <= 1` 的点，并同步清理 `mLastFrameMapPointIds_` / `mvpCurrentMapPointIds_` 里的悬空 id；
3. 建新点前先按（世界坐标 + 描述子距离）做一次近邻查重，命中就 `addObservationToMapPoint()` 而不是新建。

验收：跑完 fr1_desk2 后 `mAllMapPoints_.size()` 应稳定在数千量级且**不随帧数单调增长**；同样输入下总耗时不应随帧序号上升。

---

## 四、已确认的其它配置问题（一并处理）

| 问题 | 位置 | 说明 |
|---|---|---|
| `effective_max_iterations: 30` 无消费者 | `config/pose_graph_optimizer.yaml:169` | `config_loader.cpp:214` 只读 `max_iterations`；yaml 注释还声称"实际取 min(max_iterations, 10)"，代码里没有这句 |
| `require_candidate_agreement` 无消费者 | `config/loop_detector.yaml` | `computeCandidateConsistency()` 全仓无调用点 |
| `consistency_warning_threshold` 无消费者 | 同上 | 同上 |
| `MAX_SPATIAL_DIST_FOR_LOOP = 8.0` 写死 | `loop_detector.cpp:750` | 与 `max_spatial_dist_for_consistency`（10.0）语义重叠但取值不同 |
| `frames_between < 30` 写死 | `loop_detector.cpp:766` | 与 `min_loop_separation` 语义重叠 |
| `queryWithConsistency(..., 3, 10, 1.0)` 魔数 | `loop_detector.cpp:744` | 三个参数无名字、无出处 |
| `G2O_INCLUDE_DIRS` 指向两份 g2o | `src/sfc_core/CMakeLists.txt:58-63` | `_g2o_user_include` 与 `_g2o_types_include` 是不同安装；本次编译能过属侥幸 |
| `pose_graph_optimizer.cpp` 裸 cerr | `:931-932` 等多处 | 与 `opts_.verbose` 混用，`addLoopEdge` 成功那条无守卫 |
| SIGSEGV 防护污染业务逻辑 | `pose_graph_optimizer.cpp:16-24`、`:1369-1421` | 文件作用域 `sigjmp_buf`，`optimize()` 内 `sigsetjmp`；崩溃应该是 bug 而不是被吞掉 |

---

## 五、修复计划

按"SLEEP 顺序"：先止血（让结果不再变差），再修正确性，最后做性能与清理。每一步都可独立回滚。

### 阶段 D1 · 止血：回环不再破坏轨迹

目标：**开启 `--slam` 后结果不得差于纯 VO**。达不到就说明还没修对。

- [x] D1-a `pose_graph_optimizer.cpp`：`addLoopEdge` 返回 `bool`（或 `enum class AddEdgeResult { ADDED, DUPLICATE, VERTEX_MISSING, REJECTED }`），`loop_detector.cpp` / `tracking.cpp` 同步。 （`addLoopEdge` 已改 `bool`，10 处拒绝/去重点改为 `return false`，成功路径 `return true`）
- [x] D1-b `tracking.cpp` `detectAndCorrectLoop()`：非 `ADDED` 时直接 `return false`，不 optimize、不写回、不计数。 （`tracking.cpp` / `covis_manager.cpp` 两处调用点都在 false 时直接 return，不 optimize、不写回、不计数）
- [x] D1-c `tracking.cpp:3464-3480`：改为路线 A —— 回环**只修正关键帧位姿，不搬地图点**。 （删除"按第一个观测者 delta 搬地图点"整块，只写回关键帧位姿）
- [x] D1-d `tracking.cpp`：`detectAndCorrectLoop()` 之后补一道跳变检查（复用 `FRAME_JUMP_MAX_*`），超限则回滚本次写回。 （写回前快照全部 KF 位姿，超 `FRAME_JUMP_MAX_*` 则整体回滚并置 `tracking_ok=false`）

验收：`--slam` 的路径总长 ≤ 纯 VO 的 478 m（在 ATE 脚本就位前先用这个粗指标；就位后立刻换 R-024 口径）。

### 阶段 D2 · 回环位姿的真实尺度

- [x] D2-a `geometric_verifier.hpp`：`verifyWithEssential` 增加出参 `double* out_baseline_m`，由调用方用实测深度算中位基线传入；函数内 `T.translation = baseline * t_dir`。 （**接口形态偏离**：`verifyWithEssential` 不持有 3D 点，无法自己算深度基线，因此基线中位数在调用方 `loop_detector.cpp` 计算后乘回单位方向；`T.translation = baseline * t_dir` 的语义与计划一致）
- [x] D2-b `loop_detector.cpp:622-639`：删除 `kEMFallbackTranslationScale = 1.0`；拿不到可靠基线时 `return false`，而不是编一个 1 m。 （`kEMFallbackTranslationScale` 已删；样本 < 3 或基线非有限时 `return false`）
- [x] D2-c 帧间距离闸门（`loop_detector.cpp:644-654`）改用**匹配点世界坐标中位距离**，不再用漂移位姿的 `pose_curr.t - pose_matched.t`。 （EM 闸门改用匹配点深度实测基线中位数）

验收：日志里 `|t|` 不再恒为 1.0，且同一对关键帧的 `|t|` 与地图点实测距离一致（误差 < 20%）。

### 阶段 D3 · 阈值回到配置

- [x] D3-a `LoopDetector` 构造函数接受 `const GeometricVerifierConfig&`，删除 `loop_detector.cpp:155-172` 的全部字面量。 （**接口形态偏离**：没有改构造函数签名，而是让 `LoopDetectorConfig` 直接持有 `GeometricVerifierConfig` 成员，构造函数整份拷贝 —— 效果等价，且两条回环路径自动同步）
- [x] D3-b `config/geometric_verifier.yaml`：`min_inlier_ratio` 0.15 → 0.25，`min_inliers` 2 → 12，`min_icp_inliers` 2 → 12。 （yaml 已改为 0.25 / 12 / 12；同时把 `min_depth` 改回 0.1、`max_depth` 改回 12.0，与 Tracking 一致）
- [x] D3-c 删除 `LoopDetectorConfig` 中与几何验证重复的字段，避免两个来源（R-023）。 （`LoopDetectorConfig` 的 `ransac_iterations` / `ransac_threshold_px` / `max_icp_inlier_ratio` 已删，只留 `geometric_verifier.yaml` 一份）
- [x] D3-d `tracking.cpp` `initLoopDetection` 从 `AllConfigs` 传入 `geometric_verifier`（`test_tum_rgbd.cpp` 同步）。 （`LoopDetectorConfig` 直接持有 `GeometricVerifierConfig`，`loadAllConfigs()` 注入，两条回环路径都走它）

验收：改 `geometric_verifier.yaml` 能实际改变行为（写一个反例：`min_inlier_ratio: 0.99` 时一条回环都不该通过）。

### 阶段 D4 · 位姿图可解

- [x] D4-a 回环边按 `geometric_inliers` 加权信息矩阵，而不是固定 `loop_information_scale = 2.0`。 （新增 `loop_inlier_count_ref` / `_min_factor` / `_max_factor`，信息矩阵再乘内点计数因子）
- [x] D4-b `optimize()` 返回 `chi2` 与 `iterations` 的出参；`chi2 == 0` 视为"无有效约束"并显式告警。 （`optimize(double* out_chi2, int* out_iterations)`；chi2 == 0 时无条件告警并打印边上数量）
- [x] D4-c 里程计边也带信息权重（当前 `information_scale = 1.0` 一刀切），让回环边真的能拉动轨迹。 （`odom_info_scale_min/max` 配置化，区间由 [0.5, 2.0] 放宽到 [0.2, 4.0]）
- [x] D4-d 每 N 次回环才做一次 PGO（`loop_detection_interval` 语义），而不是每个关键帧都试。 （`loop_check_interval_kf`，默认 1 = 每个关键帧，与改动前行为一致）

验收：`loop_edge_data_` 随回环次数增长；`chi2` 非零且逐步下降。

### 阶段 D5 · 性能

- [x] D5-a `geometric_verifier.hpp` 加 `bool verbose = false` 并给全部 `std::cerr` 加守卫（用 `scripts/guard_diag_logs.js --guard=config_.verbose`，注意保留 `FATAL` / 异常类日志无条件输出，见 R-021）。 （44 处加 `config_.verbose` 守卫，2 处按 R-021 保持无条件）
- [x] D5-b `pose_graph_optimizer.cpp` 补齐裸 cerr 的守卫。 （守卫 60 处，其中 32 处故障类日志按 R-011 撤回为无条件）
- [x] D5-c `addLoopEdge` 置 `optimizerNeedsRebuild_` 只在**真的加了边**时发生（D1-a 之后自然成立）。 （`optimizerNeedsRebuild_` 现在紧邻 `return true`，只在边真的入图时置位）
- [x] D5-d `applyOptimizedPosesToFrontend()` 的 `mvpLocalKeyFrames_` 查找改成先建 `unordered_map<int, size_t>`，去掉 O(K²)。 （`local_kf_index` 预建 `unordered_map<int,size_t>`，去掉 O(K²)）
- [x] D5-e `loop_detector.cpp:751` 的 `std::min(5ul, ...)` 改用 `candidate_pool_size`；`:750` 的 `MAX_SPATIAL_DIST_FOR_LOOP`、`:766` 的 `frames_between < 30`、`:744` 的 `(3, 10, 1.0)` 全部提为具名配置项（R-018）。 （4 处硬编码提为 `dbow_temporal_window` / `dbow_max_results` / `dbow_min_distance_m` / `loop_max_spatial_dist_m` / `min_frames_between_for_spatial_gate`）

验收：640 帧总耗时 ≤ 2 分钟（约 190 ms/帧），且 D1–D4 的精度指标不回退。

### 阶段 D6 · 配置、地图点生命周期与构建清理

- [x] D6-a `effective_max_iterations`、`require_candidate_agreement`、`consistency_warning_threshold`：要么接上消费者，要么从 yaml 删除（R-022）。 （`effective_max_iterations` 已从 yaml 删除；`require_candidate_agreement` / `consistency_warning_threshold` 已接线到 `selectBestFromVerified()`）
- [x] D6-b `src/sfc_core/CMakeLists.txt:58-63`：`G2O_INCLUDE_DIRS` 只保留一份 g2o 安装。 （`G2O_INCLUDE_DIRS` 不再拼两份安装路径，改为用 g2o 导出值或从 `g2o_DIR` 反推）
- [x] D6-c `pose_graph_optimizer.cpp` 的 SIGSEGV 防护：先保留，但在修复 D1–D4 之后单独评估是否可移除（吞掉段错误会掩盖真实 bug）。 （**已删除**，与 `PLAN-loop-detection-fix.md` 的 L8 冲突时以 L8 为准，理由见执行记录）
- [x] D6-d 地图点生命周期（原因 7）：`updateLastFrame()` / 帧末的深度点改为进帧内临时池，不进 `mAllMapPoints_`；`cleanOldMapPoints()` 改为真正 `erase` 单次观测点并清理悬空 id；建点前先按世界坐标 + 描述子做近邻查重。 （临时点独立池 + `retireBadMapPoints()` 真正 erase + 体素近邻查重，见前端 E6）

验收：跑完 fr1_desk2 后 `mAllMapPoints_.size()` 不随帧数单调增长；同一输入下总耗时曲线平坦。

### 阶段 D7 · 验收（R-024）

- [ ] D7-a `scripts/evaluate_trajectory.py --align`（**不估尺度**，RGB-D 尺度来自深度）出 ATE rmse / RPE_trans rmse，覆盖 `fr1_desk2` 与 `fr1_room`。 （**未执行**：本机是 Windows、无 C++ 工具链、无 TUM 数据，无法产出 ATE/RPE。需要在 Ubuntu `/home/syf/SFC-SLAM` 上 `colcon build` 后跑 `scripts/run_and_eval_tum.sh`）
- [ ] D7-b 每个阶段结束后重跑同一口径，把数字写进本文件的"实测记录"表。 （**未执行**：本机是 Windows、无 C++ 工具链、无 TUM 数据，无法产出 ATE/RPE。需要在 Ubuntu `/home/syf/SFC-SLAM` 上 `colcon build` 后跑 `scripts/run_and_eval_tum.sh`）
- [ ] D7-c 提交信息必须写明：序列、对齐方式、ATE rmse、RPE_trans rmse（R-024）。 （**未执行**：本机是 Windows、无 C++ 工具链、无 TUM 数据，无法产出 ATE/RPE。需要在 Ubuntu `/home/syf/SFC-SLAM` 上 `colcon build` 后跑 `scripts/run_and_eval_tum.sh`）

---

## 六、验收目标（暂定，D7 就位后校准）

| 指标 | 当前 | 目标 |
|---|---|---|
| `fr1_desk2` 路径总长 | 1165.1 m | ≤ 15 m |
| ATE rmse（`--align`，不估尺度） | 未测 | ≤ 0.10 m |
| RPE_trans rmse | 未测 | ≤ 0.05 m |
| 有效回环边数 / 检出次数 | 1 / 43 | ≥ 50% |
| 640 帧总耗时 | 10+ min | ≤ 2 min |

ATE 目标值参照 ORB-SLAM3 在 `fr1_desk2` 上的公开结果（约 0.04–0.06 m），本项目允许 2–3 倍差距；超过 0.15 m 说明还有结构性错误，不应以"调参"收尾。

---

## 七、实测记录

| 日期 | 版本/改动 | 序列 | 对齐 | ATE rmse | RPE_trans rmse | 路径总长 | 耗时 |
|---|---|---|---|---|---|---|---|
| 2026-11-09 | 纯 VO（阶段 A/B1/B3 后） | fr1_desk2 | 未测 | 未测 | 未测 | 478.6 m | — |
| 2026-11-09 | B2 完整 SLAM（首次跑通） | fr1_desk2 | 未测 | 未测 | 未测 | 1165.1 m | 10+ min |

---

## 八、风险与回滚

- **D1-c（不搬地图点）会短期抬高重投影误差**，可能出现"位姿对了但地图点看着飘"的观感。判断标准是 ATE，不是肉眼。
- **D2-a 改 `verifyWithEssential` 签名**会波及 `LoopDetector` 与（已注释的）`CovisManager` 路径。改之前先 grep 全部调用点。
- **D3-b 收紧阈值会显著减少回环数**。fr1_desk2 本身回环机会就少，数量下降是预期行为，不要为了让"回环数好看"把阈值调回去。
- **项目不是 git 仓库**（`fatal: not a git repository`），改前必须另存副本：
  ```
  cp src/sfc_core/src/loopdetection/pose_graph_optimizer.cpp /tmp/pgo.base
  cp src/sfc_core/src/loopdetection/loop_detector.cpp       /tmp/loop_detector.base
  cp src/sfc_core/include/sfc_core/loopdetection/geometric_verifier.hpp /tmp/geo.base
  cp src/sfc_core/src/frontend/tracking.cpp                 /tmp/tracking.slam.base
  ```
- **本机没有 C++ 工具链**（无 cmake/g++/clang/MSVC/ninja），所有改动只能做括号配平静态检查；真实编译与运行必须在 Ubuntu 机器上由用户验证。

---

## 九、与 CODE-RULES 的关系

本文件列出的问题，每一条都能落到已有规则上，说明这些规则不是纸面约定：

| 规则 | 本次被违反的地方 |
|---|---|
| R-017（诊断输出受开关控制） | `geometric_verifier.hpp` 全文件；`pose_graph_optimizer.cpp:931-932` 等 |
| R-018（魔法数字提为具名常量） | `0.15`、`2`、`35.0`、`12.0`、`MAX_SPATIAL_DIST_FOR_LOOP=8.0`、`frames_between<30`、`(3,10,1.0)` |
| R-019（运行参数单一来源） | 几何验证阈值在 yaml 与 `loop_detector.cpp` 各一份 |
| R-022（yaml 键必须有消费者） | `effective_max_iterations`、`require_candidate_agreement`、`consistency_warning_threshold` |
| R-023（每个配置结构体唯一 yaml 归属） | `geometric_verifier.yaml` 有归属但没接线；`LoopDetectorConfig` 与 `GeometricVerifierConfig` 字段重叠 |
| R-024（精度必须用 ATE/RPE） | 第一版完整 SLAM 只看了路径总长就交付 |

建议在 D1 完成后追加一条规则：

> **R-025**：新增或修改任何图优化（PGO/BA）约束前，必须先在日志中证明该约束的平移量有真实尺度（与地图点实测距离一致），否则不得接入优化。｜理由：B2 把只有方向的 Essential 平移固定成 1.0 m 接进 PGO，导致 43 次回环全部是错误约束、轨迹膨胀到 1165 m。

---

## 十、遗留（不在本次范围）

- 15 个文件超 R-007 400 行（`tracking.cpp` 约 3680 行、`covis_manager.cpp` 2230、`pose_graph_optimizer.cpp` 1497、`loop_detector.cpp` 1321 …），按职责拆分属独立工作量。
- `test_visual_slam.cpp`（1657 行）依赖不存在的 `feature_extractor.hpp`，CMakeLists 已注释；`test_feature_pipeline.cpp` 引用早已删除的字段。两者要么修好要么删掉，不应长期留作死代码。
- `Pose` / `SE3` 类型封装（R-020）：本次多处临时拼 `toMatrix()` / 手写 `inverse()`，坐标系靠注释维护。


---

## 八、执行记录（2026-11-10）

**范围**：D1 → D2 → D3 → D4 → D5 → D6 全部落地；D7 因缺少工具链与数据集未执行。
**验证状态**：Windows 本机无 C++ 工具链，未编译；静态核对仅做到括号配对与符号残留扫描。
**与 `PLAN-loop-detection-fix.md` 的关系**：两份文件的 L/D 编号大量重叠，实际执行时以
"同一处代码只改一次"为原则合并处理，落地位置记录如下。

### D1 · 回环边真的能进图

- D1-a / D1-b：`addLoopEdge` 改 `bool`，`tracking.cpp` / `covis_manager.cpp` 两处调用点都在
  false 时直接返回。
- D1-c：`applyOptimizedPosesToFrontend()` 删掉"按第一个观测者 delta 搬地图点"整块。
  一个点被 K 个关键帧观测、各帧 delta 不同，用其一搬点等于让另外 K-1 个观测指向错误世界位置，
  地图会长出重复结构。现在只写回关键帧位姿。
- D1-d：`track()` 里 `detectAndCorrectLoop()` 前后加位姿快照，写回后位移/旋转超过
  `FRAME_JUMP_MAX_TRANSLATION_M`(0.5) / `FRAME_JUMP_MAX_ROTATION_RAD`(0.524) 时整体回滚
  （`mAllKeyFrames_` / `mvpLocalKeyFrames_` / `mpMap_` / `loop_detector_` / 当前帧 / 上一帧），
  并置 `tracking_ok = false`。

### D2 · 真实尺度

- D2-a（**接口形态偏离**）：`verifyWithEssential` 只接收 2D 关键点，拿不到深度，无法自己算基线。
  实现改为：调用方用 EM 解出的旋转对 3D 对应点求 `t = p_train - R_em * p_query`，
  投影到单位方向取中位数，再 `T_em.block<3,1>(0,3) = baseline * t_dir`。
  语义与计划一致（平移长度来自深度实测），只是计算位置在调用方。
- D2-b：`kEMFallbackTranslationScale` 已删；样本 < 3 对或基线非有限时 `return false`。
- D2-c：EM 的帧间距离闸门改用同一批样本的模长中位数，不再用 `pose_curr.t - pose_matched.t`
  —— 回环场景里这两帧的 PnP 结果恰恰漂得最厉害。

### D3 · 阈值回配置

- D3-a（**接口形态偏离**）：没有改构造函数签名，改为 `LoopDetectorConfig` 直接持有
  `GeometricVerifierConfig` 成员，构造函数整份拷贝。效果等价，且 CovisManager 与 Tracking
  两条回环路径自动同步。
- D3-b：`config/geometric_verifier.yaml` 收紧 `min_inliers` 12 / `min_inlier_ratio` 0.25 /
  `min_icp_inliers` 12；**同时修正 `min_depth` 0.05→0.1、`max_depth` 30→12**
  （接上消费者之后这份 yaml 才真正生效，而它与 Tracking 的深度范围原本就不一致）。
- D3-c：`LoopDetectorConfig` 的 `ransac_iterations` / `ransac_threshold_px` /
  `max_icp_inlier_ratio` 删除，只留几何验证器那一份；`config/loop_detector.yaml` 对应条目
  改为指向 `geometric_verifier.yaml` 的注释。
- D3-d：`loadAllConfigs()` 里 `all_cfg.loop_detector.geometric_verifier = all_cfg.geometric_verifier;`，
  `initLoopDetection` 整份传入，测试无需改签名。

### D4 · 位姿图可解

- D4-a：回环边信息矩阵再乘"内点计数因子" `inliers / loop_inlier_count_ref`，夹在
  `[loop_inlier_count_min_factor, loop_inlier_count_max_factor]` = `[0.25, 4.0]`。
  原来只有内点率的一次项，"2 对全内点"与"200 对全内点"权重完全相同。
- D4-b：`optimize(double* out_chi2, int* out_iterations)`；`chi2 == 0` 时无条件告警并打印
  里程计边数 / 回环边数 / 活跃边数。两处调用点都把 chi2 打进日志。
- D4-c：`odom_info_scale_min/max` 配置化，区间由写死的 `[0.5, 2.0]` 放宽到 `[0.2, 4.0]`。
  原区间把 30% 与 90% 内点的里程计边压成同一权重，回环修正传不出去。
- D4-d：新增 `loop_check_interval_kf`（默认 1 = 每个关键帧检测，与改动前行为完全一致）。
  原来唯一的闸门是"同一个 KF 不重复进来"，等于每个关键帧都跑一次 BoW + 几何验证 + Sim3/ICP + 全局 PGO。

### D5 · 性能

- D5-a：`geometric_verifier.hpp` 44 处加 `config_.verbose` 守卫（2 处按 R-021 无条件保留）。
- D5-b：`pose_graph_optimizer.cpp` 60 处加 `opts_.verbose` 守卫，其中 32 处故障类日志
  （拒绝/失败/NaN/异常/initializeOptimization 失败）按 R-011 撤回为无条件输出。
- D5-c：`optimizerNeedsRebuild_ = true;` 现在紧邻 `return true;`，只在边真的入图时置位。
- D5-d：`applyOptimizedPosesToFrontend()` 预建 `unordered_map<int,size_t> local_kf_index`，
  去掉 K×K 线性查找。
- D5-e：`min(5ul, ...)` → `candidate_pool_size`；`MAX_SPATIAL_DIST_FOR_LOOP = 8.0`、
  `frames_between < 30`、`queryWithConsistency(..., 3, 10, 1.0)` 全部提为
  `dbow_temporal_window` / `dbow_max_results` / `dbow_min_distance_m` /
  `loop_max_spatial_dist_m` / `min_frames_between_for_spatial_gate`。

### D6 · 配置、地图点生命周期与构建

- D6-a：`effective_max_iterations` 从 `config/pose_graph_optimizer.yaml` 删除 ——
  全仓没有任何代码读取它，改它不会有任何效果；同一段注释还声称"实际代码中取
  min(max_iterations, 10)"，而 `pose_graph_optimizer.cpp` 里是把 `opts_.max_iterations`
  原样传给 g2o。两处注释一并更正。`require_candidate_agreement` /
  `consistency_warning_threshold` 选择"接线"而不是删除（见 D9/L9）。
- D6-b：`src/sfc_core/CMakeLists.txt` 不再把 `/home/syf/g2o_user_install/include` 与
  `/home/syf/g2o/build/install/include` 两份安装同时拼进 `G2O_INCLUDE_DIRS`
  （两份头文件版本不同时，谁赢取决于书写顺序）。改为优先用 `find_package(g2o)` 的导出值，
  为空时从 `g2o_DIR/../../..` 反推 `<prefix>/include`，并 STATUS 打印来源。
  `ResolveThirdparty.cmake` 已经把 `g2o_DIR` 唯一化，这里不再造第二个来源。
- D6-c：**已删除 SIGSEGV 防护**，而不是"先保留"。原因是 `PLAN-loop-detection-fix.md` 的 L8
  独立判定它有害，且两条理由可以同时成立：段错误时 g2o 已处于半更新状态，`longjmp` 回来的是
  损坏的图 —— 把"崩溃"换成"静默错误轨迹"。计划 D6-c 写"修复 D1–D4 之后单独评估"，
  评估结论是删除，因此执行了删除并保留 C++ `try/catch`。
- D6-d：临时 VO 点改为进独立池 `mTemporalMapPoints_`；`retireBadMapPoints()` 真正
  `erase` 坏点并清理悬空 id；`createMapPoint()` 先做体素近邻查重（0.10m 体素 / 0.05m 判重）。

### D7 · 未执行

本机无 C++ 工具链、无 TUM 数据集，`scripts/evaluate_trajectory.py` 与
`scripts/run_and_eval_tum.sh` 无法运行，**没有任何 ATE / RPE 数字**。
需要在 Ubuntu `/home/syf/SFC-SLAM` 上 `colcon build` 后按下面口径跑：

```bash
bash scripts/run_and_eval_tum.sh fr1_desk2   # 纯 VO
bash scripts/run_and_eval_tum.sh fr1_room
python3 scripts/evaluate_trajectory.py --align --no-scale <traj> <gt>
```

### 未完成 / 需确认

1. **编译**：全部改动未编译。`std::clamp`（E5）依赖 C++17；`<cstdint>`（E6 体素键）、
   `<cmath>`（EM 基线）已补，其余头文件依赖未逐一验证。
2. **D7 的量化验收**：上面的基线（纯 VO 478.569 m / 完整 SLAM 1165.124 m / 真值 5–12 m）
   是改动前的数字，改动后必须重测才有意义。
3. `config/covis_manager.yaml` 的 `loop_detection_interval: 9999` 未动 —— 它等于关闭
   CovisManager 回环路径，与 `PLAN-tracking-drift-fix.md` 的 B2 是同一个待决策问题。
4. D1-c 之后地图点不再随回环写回搬动，地图需要若干帧 `poseOptimization` 才能重新收敛；
   真实序列上应观察回环后短期内的 reprojection error 是否先升后降。
