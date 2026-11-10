# 视觉 SLAM 轨迹漂移诊断与整改计划

> 日期: 2026-11-09
> 目标: 修复 TUM-RGBD 测试集轨迹与 groundtruth 差距过大（当前 fr1_desk2 路径总长 478.6m vs 真实约 5–12m），并同步提升代码可读性。
> 依据: 通读 `src/sfc_core` 核心链路（tracking / robust_pnp_solver / covis_manager / config）+ 逐帧诊断日志 `src/diag_log2.txt`。

---

## 一、诊断结论（根因清单，按严重程度排序）

### P0-1 · poseOptimization 外点剔除失效，位姿优化发散（最直接根因）

**证据**: `src/diag_log2.txt` 每帧 `poseOpt-DIST` 的 `init_reproj_err_avg` 从初始化帧 f=0 的 10px 一路飙升：f=1 为 64px、f=19 为 121px、f=23 为 728px、f=31 为 584px、f=49 为 3022px（max 达 22834px）。重投影误差本应收敛到亚像素级，却呈指数发散。

**代码位置**: `src/sfc_core/src/frontend/tracking.cpp` 的 `poseOptimization()`（约 2208–2698 行）：

1. 第 0/1 轮迭代 `chi2_th_this_round = 10000.0f`（ORB-SLAM3 标准是 5.99），前两轮几乎不剔外点；
2. 第 1 轮迭代 `e->setRobustKernel(nullptr)` 移除了 Huber 核——注释标成"修复"，实际适得其反：无核后外点的二次代价主导 LM 优化，把位姿拉向错误解；
3. 尾部 `nInliers < pose_opt_min_inliers`（默认 5）仅告警不拒绝，坏位姿仍被采纳。

**结论**: 位姿优化在大量外点上发散是 478m 轨迹的直接来源。

### P0-2 · relocalization 接受坏解，反复注入错误位姿

**证据**: 约 frame 610 起进入 LOST→RELOC 死循环，每帧强制建 KF（`FORCE KF after RELOC`）；Reloc PnP 内点仅 8–14 个、ratio 0.07–0.2 仍被接受；f=611 单帧跳 7.27m。

**代码位置**: `tracking.cpp` 的 `relocalization()`（约 1476–2012 行），接受阈值渐进放宽到 `reloc_min_inliers=10`、`min_ratio=0.05`，`robust_pnp_solver.cpp` 在 <50 点时 ransac 阈值 ×2.0。

**结论**: 前端一旦丢失，低质量 reloc 位姿被无条件采纳并强制建关键帧，把坏解固化进地图。

### P1-3 · 深度反投影缺防护（诊断日志中的 `±1e10` 是哨兵值，不是地图污染）

**证据**: f=35/39/43/45/47/53 的日志 `depth=[10000000000.000, -10000000000.000] avg=0.000`。

**核实结论（修正初版判断）**: `±1e10` 是**诊断打印的哨兵值**，不是地图点里的真实数据 —— 打印前的统计只在 `z > 0 && z < 10` 时更新 min/max，计数器为 0 时保留初值。它真正的含义是"这一帧所有点都落在相机后方"，也就是位姿已经翻过去了（P0-1 的后果），而不是"地图点被 NaN 污染"。`avg=0.000` 同理是空统计。

**仍然存在的真实缺口**: `createMapPoint()` 不校验 `world_pt` 是否有限。位姿翻车时 `1/z` 爆炸产生的 Inf/NaN 会被原样写进地图，后续所有投影都以它为输入。

### P1-4 · 配置多源不一致

| 参数 | tracking.yaml | tracking.hpp 默认 | test_tum_rgbd.cpp | 影响 |
|------|---------------|-------------------|-------------------|------|
| max_depth | 12.0 | 10.0 | 5.0 | 三处不一致，深度过滤口径混乱 |
| n_features | 1500 | 2000（OctreeORBConfig） | 2000 | 提取数量不一致 |
| depth_edge_thresh | 0.2 | 0.25 | 0.15 | 深度边缘剔除口径混乱 |
| forced_keyframe_interval | 30 | 6 | 20 | 关键帧密度不可预期 |
| min_keyframe_translation | 0.15 | 0.05 | 0.10 | 同上 |

> **阶段 B1 执行中修正的一处误判**：初版诊断把 `enable_local_ba` / `enable_pgo` 记作"三处冲突"。核实后实为**死字段**——`TrackingConfig` 里的这两个字段在 `tracking.cpp` 中只有 `:402-403` 两行日志打印，没有任何逻辑分支读取；真正生效的是 `CovisConfig` 的同名字段（消费者在 `covis_manager.cpp:673/677/905/190/225/1513`），而纯 VO 路径根本没有 `CovisManager` 实例（`tracking.hpp:149 getCovisManager()` 恒返回 `nullptr`）。所以正确的修法不是"统一成一个值"，而是**删掉这两份假开关**。

### P1-5 · 纯 VO 模式无回环校正，发散无兜底

`enable_pgo=false`、`covis_manager.yaml` 中 `enable_loop_detection=false`、`loop_detection_interval=9999`，前端任何发散都会累积，无回环/PGO 兜底。这是 478m 被"放大"而非"修正"的系统性原因。

### P2-6 · 位姿帧间跳变异常

f=0(0,0,0.02) → f=5(1.157,0.008,0.610) → f=6(-0.847,-0.383,-0.631) → f=8(0.489,0,-0.895)，单帧跳 0.5–1m（真实应 3–7cm/帧），由 P0-1/P0-2 叠加导致。

---

## 二、修复计划（分阶段，每阶段可独立验证）

### 阶段 A · 阻断发散（P0，先止血，1–2 天）

> **状态（2026-11-09 执行完毕，代码已落地）**
> 改动文件：`src/sfc_core/src/frontend/tracking.cpp`（3342 → 3354 行）、`src/sfc_core/src/frontend/robust_pnp_solver.cpp`。
> 新增工具：`scripts/check_braces.js`（括号配平静态检查）、`scripts/guard_diag_logs.js`（日志开关守卫批处理）。
> ⚠️ **本机无 C++ 工具链（无 cmake/g++/clang/MSVC/ninja），阶段 A 只做了静态验证，未编译、未跑数据集。**
> 补跑验收前，任何"已修复"结论都只停留在代码层。

- [x] A1. 恢复 `poseOptimization` 外点剔除标准（`tracking.cpp`）：
  - 四轮统一 `constexpr float CHI2_THRESHOLD_2DOF_95 = 5.99f;`，删除原 `it==0/1 → 10000.0f` 的宽松分支；
  - 删除 `if (it == 1) e->setRobustKernel(nullptr);`，全程保留 Huber（`delta = 5.0`）；
  - 外点剔除从 `if (it >= 1)` 放宽改为**每轮无条件** `if (e->chi2() > th) e->setLevel(1);`
  - 尾部改为**先判后写**：`if (nInliers < config_.pose_opt_min_inliers) { return 0; } mCurrentFrame_->pose = new_pose;` —— 坏位姿不再污染 `mCurrentFrame_->pose`。调用方 `tracking.cpp:704 / 1003 / 1173` 已确认会把 0 正确传播为跟踪失败。
- [x] A2. 收紧 `relocalization` 接受阈值：
  - `reloc_min_inliers = max(30 - n_fail/5, 20)`（原 `max(15 - n_fail/5, 10)`）；
  - `reloc_min_ratio = max(0.25 - (n_fail/10)*0.02, 0.15)`（原 `max(0.10 - ..., 0.05)`）；
  - **关键**：接受条件从 `(inlier_ratio >= ratio || reasonable_delta)` 改为全 AND —— 原 `reasonable_delta` 默认 true，只要单帧位移 <3m 就绕过内点率，日志里 ratio=0.07 的 reloc 正是由此进图；
  - 宽窗口复核分支与主路径同源（`inliers >= reloc_min_inliers && wide_ratio >= reloc_min_ratio && wide_delta_reasonable`），原为 `>=15 && >=0.10 && z>0.1`；
  - `robust_pnp_solver.cpp` 稀疏点阈值放宽 `×2.0 → ×1.2`（`RANSAC_THRESH_SCALE_SPARSE`）、`×1.5 → ×1.1`（`RANSAC_THRESH_SCALE_MEDIUM`）。
  - 顺带修掉原有**编译错误级**结构缺陷：宽窗口搜索块缺少 `for (const auto& [idx_curr, p3d_world] : additional_pts)` 循环开头，`continue;` 与 `idx_curr` 悬空。
- [x] A3. 全链路深度投影防护：
  - `createMapPoint()` 入口新增 `if (!world_pt.allFinite() || world_pt.norm() < 1e-6) return -1;`，8 个调用点各加 `if (mp_id < 0) continue;`；
  - 新增具名常量 `MIN_PROJECTION_Z = 0.05`、`RELOC_MAX_DELTA_T = 3.0`、`MIN_RELOC_CAMERA_Z = -2.0`；
  - 核查结论：全文件约 30 处 `1/z` 投影点原本已有 `z < 0.05 || !std::isfinite(z)` 防护，无需重复添加。
- [x] A4（附带）. 日志开关落实 R-017：`tracking.cpp` 39 处 `std::cerr` 全部纳入 `config_.verbose_diag`；`robust_pnp_solver.cpp` 症状类日志纳入 `config_.verbose`，**但 `FATAL`/`exception`/`错误：`/`点数不足` 等 14 处错误日志保持无条件输出**（R-011/R-012 要求错误必须可见，不能被 verbose 静音）。

**静态验证**: `node scripts/check_braces.js <file>` 对两个文件均返回 `brace=0 paren=0 brack=0`。
**待补验收（有工具链的机器上执行）**: 编译 → 重跑 fr1_desk2 → `init_reproj_err_avg` 应稳定在个位数 px，且无 `depth=[±1e10]` 哨兵日志、无 LOST→RELOC 死循环。

### 阶段 B · 一致性收敛（P1，1–2 天）

> **状态（2026-11-09 执行完毕，B1/B3 代码与配置已落地；B2 待决策）**
> 改动文件：`src/sfc_core/include/sfc_core/config/config_loader.hpp`、`src/sfc_core/src/config/config_loader.cpp`、`src/sfc_core/include/sfc_core/frontend/tracking.hpp`、`src/sfc_core/src/frontend/tracking.cpp`、`src/sfc_core/test/test_tum_rgbd.cpp`、`src/sfc_core/test/test_tum_module_diag.cpp`；配置文件 `config/tracking.yaml`、`config/feature_extractor.yaml`、`config/feature_matcher.yaml` 重写，新增 `config/camera_fr1.yaml`、`config/camera_fr2.yaml`、`config/camera_fr3.yaml`。
> ⚠️ 同样只有静态验证（括号配平 + grep 覆盖率），未编译。

- [x] B1. 配置收敛到单一来源（R-019）。已落地的四类修正：

  **B1-a 删除假开关**。`TrackingConfig::enable_pgo` / `enable_local_ba` 两个死字段删除（`tracking.hpp`），`[TRACK-START]` 打印同步改为输出真正生效的 `max_depth` / `n_features` / `disable_undistort`。`loadTrackingConfig` 里对应两行 `getBool` 一并删除。真正的开关留在 `CovisConfig`（`covis_manager.yaml`）与 `PoseGraphOptimizerOptions`（`pose_graph_optimizer.yaml`）。

  **B1-b 修掉静默失效的配置文件路径**。`config_loader.cpp` 的 `loadAllConfigs` 一直读 `octree_orb.yaml`，而仓库里从来没有这个文件 —— 每次启动打 WARNING 后静默用结构体默认值，`config/feature_extractor.yaml` 形同废纸。已改为读 `feature_extractor.yaml`。

  **B1-c 修掉 yaml 键名与 loader 键名错配**。新增 `getIntAny` / `getDoubleAny` / `getBoolAny` 三个别名读取函数：`fast_threshold` ↔ `ini_th_fast`、`min_fast_threshold` ↔ `min_th_fast`、`max_hamming_dist` ↔ `max_match_dist`。同时补齐 `loadOctreeORBConfig` 缺失的 `edge_threshold` / `patch_size`，以及 `loadFeatureMatcherConfig` 缺失的 `cross_check` / `th_low` / `th_high` / `histo_length` / `check_orientation`（这 5 个字段此前完全没有加载路径，只能吃结构体默认值）。

  **B1-d 相机内参搬到 config 层**。新增 `struct CameraIntrinsicsConfig` + `loadCameraIntrinsicsConfig()`（`config_loader.hpp/.cpp`），配套 `config/camera_fr1.yaml` / `camera_fr2.yaml` / `camera_fr3.yaml`（值取自 ORB_SLAM3 的 TUM1/TUM2/TUM3.yaml）。`loadTrackingConfig` 里那份硬编码 fr2 的 `camera_fx..camera_d4` 加载整段删除 —— 那正是"yaml 写 fr1、实际跑 fr2"这类事故的来源。`config/tracking.yaml` 第十节同步删除 9 个 camera_* 键。

  **B1-e 测试程序不再硬编码**。`test_tum_rgbd.cpp` 删掉整个 `loadTUMIntrinsics()`（内部写死 fr1 内参）与整块 `tcfg.* = 字面量` 配置，改为 `loadAllConfigs(args.config_dir)` + `resolveIntrinsics(args.camera_yaml, ...)`，命令行只在显式给出时覆盖（`0` / 负值代表"用 yaml"）。`test_tum_module_diag.cpp` 同构改造，`RobustPnPConfig` 与 `MatcherConfig` 也改为从 yaml 取，只保留 `expansion_rounds = 0` 这一处显式偏离并写明理由。

  **B1-f 唯一合并点**。`TrackingConfig` 内嵌的 `extractor_config` / `matcher_config` 只允许在 `loadAllConfigs()` 末尾从 `feature_extractor.yaml` / `feature_matcher.yaml` 合并一次；`loadTrackingConfig` 里原先那份重复读取已删除（`tracking.yaml` 第二/三节整段移除）。

  **B1-g 清理无消费者 yaml 键**。删除 `enable_uniform` / `grid_cols` / `grid_rows` / `max_per_grid`（`OctreeORBConfig` 无对应字段，实现是四叉树不是网格）、`matcher_type` / `min_dist_slack` / `bidirectional_match` / `skip_bidirectional` / `ransac_threshold`（`MatcherConfig` 无对应字段，匹配器固定 `BFMatcher(NORM_HAMMING)`）。

  **B1-h 关键帧标记改用真实判定**。`test_tum_rgbd.cpp:604` 原来用 `i % tcfg.forced_keyframe_interval == 0` 推测关键帧，与 `needNewKeyFrame()` 的实际决策无关，会让 `_kf` 轨迹与真实地图内容脱节。新增 `Tracking::isLastFrameKeyFrame()` 与成员 `mbLastFrameIsKeyFrame_`（`tracking.cpp:747` 写入），测试端改用它。

- [ ] B2. 明确纯 VO 定位（**待决策**）：若目标就是纯 VO，则关闭 covis_manager 的 local_ba 与 loop 残留；若目标是完整 SLAM，开启 `enable_pgo` + `enable_loop_detection`，让回环/PGO 兜底。
  > 补充事实：`Tracking::initLoopDetection()` 目前是**空实现**（`tracking.cpp:3205-3208` 函数体 `{}`），`getLastBAMapPointUpdates()` 返回空，`checkKeyFrame()` / `checkKeyFrameSimple()` 都直接 `return needNewKeyFrame()` 忽略传参。也就是说"切到完整 SLAM"不是改配置就能完成，还要把这条链路补起来。
- [x] B3. 加帧间位姿跳变保护。`tracking.cpp` 在 `poseOptimization()` 返回后新增一道独立检查：拿 `mCurrentFrame_->pose` 与 `mLastFrame_->pose` 比，超过 `FRAME_JUMP_MAX_TRANSLATION_M = 0.5`（米）或 `FRAME_JUMP_MAX_ROTATION_RAD = 0.524`（≈30°）就置 `nGood = 0` → `tracking_ok = false`，走原有的"沿用上一帧位姿 + 清速度"路径，而不是把跳变累积进轨迹。
  - 为什么需要这一层：`poseOptimization()` 内部已有两道检查（`t_diff > 1.0m / rot_diff > 1.0rad` 相对本帧初值、`MOTION_T_DIFF_TH = 0.50m` 相对运动模型预测），但它们都以**本帧初值**为参照系 —— 初值本身错了就全盘失效（诊断日志里 f=5→f=6 从 (1.157, 0.008, 0.610) 跳到 (-0.847, -0.383, -0.631) 正是这种情况）。新增的这道以**上一帧真实位姿**为参照，是唯一不依赖初值的关卡。
  - `track_method == "RELOC"` 显式豁免：重定位本来就是"跳回正确位置"，与上一帧比必然很大，由 `relocalization()` 自己的内点率与 `RELOC_MAX_DELTA_T` 门槛负责。
  - 阈值取 0.5m 而非 0.1m：TUM 手持 30Hz 单帧约 3–7cm，留一个数量级余量，只拦"明显不是单帧运动"的解，避免把正常快速平移误判成丢失（收紧到 0.1m 会让 fr1_desk 这类快速扫视序列频繁丢跟踪）。

**验收**: 配置无冲突告警；帧间位移恢复到 cm 级。

### 阶段 C · 精度对标（P2，持续）

> **状态（2026-11-09）**：C1 已落地并在合成轨迹上验证；C2 缺数据集与工具链，未执行。

- [x] C1. 建立 ATE/RPE 自动评测。
  - `scripts/evaluate_trajectory.py`（新）：读 TUM 格式轨迹 → 时间戳关联（双指针，容差默认 0.02s）→ Umeyama 对齐（可选估尺度）→ 输出 ATE(rmse/mean/median/std/min/max)、RPE_trans、RPE_rot(deg)、est/ref 单帧位移中位与最大值。只用 numpy，不依赖 evo。
  - `scripts/run_and_eval_tum.sh`（新）：一次跑完「执行 SLAM → 写轨迹 → 对齐 → 评测」，产物落在 `eval_out/`（`_traj.tum` / `_aligned.tum` / `_run.log`）。
  - **为什么必须换口径**：旧的验收标准是"看日志里的路径总长"，而路径总长对单帧纬度跳变极其敏感——fr1_desk2 报出 478m 而真实约 5–12m，这个数字既不能判断位姿对错，也无法跨次比较。ATE/RPE 才是可比较的量。
  - 验证方式（本机可做，不需要 C++ 工具链）：用 numpy 合成 4 组 fixture —— 完全一致（ATE=0）、刚体+平移（不对齐 ATE=3.85m / 对齐后 0）、5cm 噪声（ATE rmse=0.0789m，单帧位移中位 0.1077m vs 真值 0.0403m）、时间基错位 100s（配对 0 帧，退出码 2）。四组结果都符合解析预期。
  - 一个刻意的取舍：`--align` 默认**不估尺度**（`--align-scale` 需显式开启）。RGB-D 的尺度由深度图给定，对齐时再估尺度会把尺度错误算进对齐变换里藏起来。
- [ ] C2. 以 fr1_desk2 / fr1_room 为基准，定位残留漂移来源（初始化尺度、关键帧策略、描述子阈值）。
  > 阻塞原因：本机既无 C++ 工具链也无 TUM 数据集，无法产生真实轨迹。有环境后按 `scripts/run_and_eval_tum.sh` 跑，先看 ATE rmse 与 `[step]` 行的单帧位移是否落回 cm 级，再逐项试关键帧策略与描述子阈值。

**验收**: ATE 达到可比对量级（目标先回到 1m 内，再谈 cm 级）。

---

## 三、可读性整改计划（与规范更新配套）

| 序号 | 问题 | 现状 | 整改 |
|------|------|------|------|
| 1 | 单文件超限 | 15 个文件 >400 行，tracking.cpp 3342 行 | 按职责拆分（tracking → 初始化/跟踪/重定位/位姿优化/关键帧） |
| 2 | 无条件调试输出 | 海量 `std::cerr`，`verbose_diag` 形同虚设 | 引入统一日志宏，受 `verbose_diag` 控制 |
| 3 | 叙事性注释 | "★★★ Bug Fix" 泛滥，复述"做了什么" | 改为"为什么"（R-010 精神） |
| 4 | 魔法数字 | th=20/30、radius=15/20/30、chi2=10000、hamming≤100 等 | 提取命名常量并标注单位与来源 |
| 5 | 坐标系靠注释 | Pose/变换约定仅注释维护 | 用类型封装（SE3/Pose 类 + 显式变换函数） |
| 6 | 重复代码 | KeyFrameCacheEntry 填充 3+ 处几乎相同 | 抽 `fillKeyFrameCacheEntry()` |
| 7 | 命名不一致 | `mnMatchesInliers_` 匈牙利+驼峰混用 | 统一 camelCase（R-004） |

---

## 四、风险与回滚

- 阶段 A 收紧阈值可能降低召回（跟踪更易丢），需用日志观察跟踪成功率，避免"修好了发散、修坏了跟踪"。
- 所有改动保留回滚副本；每阶段跑一次测试并记录 ATE/RPE 对比。

### 回滚清单（阶段 A 执行后补充）

| 文件 | 备份路径 | 说明 |
|------|----------|------|
| `src/sfc_core/src/frontend/tracking.cpp` | `/tmp/tracking.base` | 日志守卫施加前、poseOptimization 改动后 |
| `src/sfc_core/src/frontend/tracking.cpp` | `/tmp/tracking.cpp.bak` | 更早一轮（阶段 A 改动前） |
| `src/sfc_core/src/frontend/robust_pnp_solver.cpp` | `/tmp/pnp.base` | 阶段 A 改动前原件 |

> 注意：本项目**不是 git 仓库**（`git status` 报 `not a git repository`），无法用 `git checkout` 回滚。后续接入版本管理前，所有回滚只能依赖上述副本。
> `/tmp` 在 Windows Git Bash 下重启可能被清理，**建议尽快把这三份副本另存到仓库外的持久目录**。
> 阶段 B 的改动以配置与测试程序为主，未另存副本 —— 若需回退，`config/*.yaml` 的旧版本可从本文件的"B1 执行记录"逐条还原，代码侧只需还原 `config_loader.*`、`tracking.hpp` 的公开接口与两个 test。

---

## 五、配置归属（B1 完成后的唯一来源地图）

后续调参只改这张表里的文件，不要在其他地方再写一份同名字段（R-019）。

| 配置结构体 | 唯一来源 | 消费者 |
|---|---|---|
| `OctreeORBConfig` | `config/feature_extractor.yaml` | `src/sfc_core/src/feature/octree_orb.cpp` |
| `MatcherConfig` | `config/feature_matcher.yaml` | `src/sfc_core/src/feature/feature_matcher.cpp`、`tracking.cpp:1376` |
| `TrackingConfig`（顶层字段） | `config/tracking.yaml` | `src/sfc_core/src/frontend/tracking.cpp` |
| `CameraIntrinsicsConfig` | `config/camera_fr1.yaml` / `camera_fr2.yaml` / `camera_fr3.yaml` | 由调用方 `makeK()` / `makeD()` 注入 `TrackingConfig` |
| `CovisConfig` | `config/covis_manager.yaml` | `src/sfc_core/src/frontend/covis_manager.cpp` |
| `RobustPnPConfig` | `config/robust_pnp_solver.yaml` | `src/sfc_core/src/frontend/robust_pnp_solver.cpp` |
| `PoseGraphOptimizerOptions` | `config/pose_graph_optimizer.yaml` | `src/sfc_core/src/loopdetection/pose_graph_optimizer.cpp` |
| `LoopDetectorConfig` | `config/loop_detector.yaml` | `src/sfc_core/src/loopdetection/loop_detector.cpp` |
| `PlaceRecognitionConfig` | `config/place_recognition.yaml` | `src/sfc_core/src/loopdetection/place_recognition.hpp` |
| `GeometricVerifierConfig` | `config/geometric_verifier.yaml` | `src/sfc_core/include/sfc_core/loopdetection/geometric_verifier.hpp` |

`TrackingConfig::extractor_config` / `matcher_config` 是内嵌副本，只在 `loadAllConfigs()` 末尾从上面两个文件合并一次，**不得**在 `loadTrackingConfig` 里再读一遍。

---

## 六、执行环境说明（重要）

本机**没有安装任何 C++ 工具链**：

- `cmake` / `g++` / `gcc` / `clang` / `make` / `ninja` / `cl.exe` 均不存在；
- 无 Visual Studio、无 MSVC 生成器、无 vcpkg、无 LLVM、无 WSL 发行版；
- 可用工具：Node.js（v26.13.0）、Python 3（带 numpy 2.3.5，DeepSeek Harness 自带运行时）。

因此阶段 A / B 的验证边界是**静态检查**（括号配平、grep 覆盖率、调用点语义闭合），不含编译与数据集回归；阶段 C 的评测脚本用 numpy 合成轨迹做了数值验证，但没有跑过真实序列。

在有工具链的环境上按顺序补：

1. `cd src/sfc_core && cmake -S . -B build && cmake --build build -j`（先解决可能的编译错误——本轮改动只在静态层面验证过）；
2. 用 `scripts/run_and_eval_tum.sh <dataset> config/camera_fr1.yaml` 跑 fr1_desk2，得到 ATE/RPE 基线；
3. 对照 §一 的四条根因逐项确认是否真的被消除（重投影误差应落回个位数 px、无 `depth=[±1e10]` 哨兵日志、无 LOST→RELOC 死循环、单帧位移落回 cm 级）。


---

## 七、执行记录（2026-11-10）

**状态**：A1–A4、B1、B3、C1 的代码在此之前已在位（本文件已勾选）。本次执行另外两份
计划（`PLAN-frontend-pose-fix.md` / `PLAN-loop-detection-pgo-fix.md`）时，有若干改动
直接落在这个文件的根因上，一并记录，避免以后把同一处修复当成两件事。

**验证状态**：Windows 本机无 C++ 工具链、无 TUM 数据集，**未编译、未跑评测**。
C1 的脚本存在但从未实跑，因此本文件里没有任何 ATE/RPE 数字。

### 本次与 P0-1（poseOptimization 外点剔除）相关的改动

- `poseOptimization()` 的 g2o 顶点现在显式从 `T_wc` 转成 `T_cw`（`getCwFromPose`），
  取回后再转回 `T_wc`（`poseFromCw`）。在此之前，`Pose` 被当作 `T_wc` 使用，
  却原样喂给按 `T_cw` 解算的 `VertexSE3Expmap` + `EdgeSE3ProjectXYZ` ——
  **A1 恢复的外点剔除标准，是建立在一组位姿约定错误的残差上的**。
  两者现在一致，A1 的 `CHI2_THRESHOLD_2DOF_95 = 5.99` 才真正作用在正确的误差上。
- `RobustPnPSolver` 的对外出口统一返回 `T_wc`（内部一律 `T_cw`，只在出入口各转一次）。
- 观测权重改为按金字塔层加权（`invSigma2 = 1/(1.2·scale^level)^2`），原来固定 `1.2`，
  等于把高层模糊特征和零层清晰特征当成同等可信。

### 本次与 P1-5（纯 VO 无兜底）相关的改动

- `config/covis_manager.yaml:178` 的 `loop_detection_interval: 9999` **仍未动**，
  即 CovisManager 那条回环路径依然实际关闭。B2 仍是待决策项（纯 VO 还是完整 SLAM），
  本次不擅自改变产品行为。
- 但原来"即使开了回环也不会生效"的两处 bug 已修：`addOdometryEdge()` 的
  `setLevel(1)`（把全部里程计边移出优化）+ `initializeOptimization()` 默认 level 0，
  合起来让位姿图只剩 1 条回环边、chi2 恒 0；以及 `geometricVerification()` 的
  Sim3 输入被两次 world→camera 变换（点云被搬到相机背后，covis 路径永远失败）。
  也就是说：**B2 一旦决定走完整 SLAM，兜底路径这次是真的能工作了**。

### 本次与 P2-6（帧间跳变）相关的改动

- B3 的帧间跳变保护扩展到回环写回：`track()` 里 `detectAndCorrectLoop()` 前后快照
  全部关键帧位姿，写回后位移 > 0.5 m 或旋转 > 0.524 rad 则整体回滚并置
  `tracking_ok = false`。原来全局 PGO 的写回没有任何幅度检查，
  而位姿图这次开始真的会移动顶点。
- `updateMotionModel()` 的合理性阈值 1.0 m/帧 / 45° → **0.15 m/帧 / 30°**
  （室内 30 Hz 手持每帧真实位移 3–7 cm，原值等于对恒速模型无约束）。
- `updateLastFrame()` 删除"用恒速外推覆盖上一帧已优化位姿"的整块 —— 那是漂移的直接来源之一：
  每帧丢一次优化结果，误差按帧累积。
- `trackWithMotionModel()` 失败时把位姿退回本帧起点，不再把恒速预测的坏初值留给
  `trackReferenceKeyFrame()` / 重定位。
- 几何一致性容差 `max(radius*2.5, 20px)` → `max(radius*1.5, 8px)`；
  PnP 与上一帧的一致性阈值 5.0 m / 1.5 rad → 0.3 m / 0.3 rad（原值等于"PnP 永远通过"）。

### 仍未做

1. **B2 待决策**（`loop_detection_interval: 9999` 是否保留）——本次刻意不动。
2. **C2 未做**：没有 ATE/RPE，无法定位残留漂移来源。
3. **编译**：所有改动未经编译。C++17 已确认（`src/sfc_core/CMakeLists.txt:20`），
   `std::clamp`（观测权重）与 `<algorithm>` 依赖均已满足。
4. 阶段 A 的回滚清单仍为空 —— 项目不是 git 仓库，回滚依赖备份副本，
   建议在有工具链的机器上先把当前状态打一个 tag 或整目录快照再编译。
