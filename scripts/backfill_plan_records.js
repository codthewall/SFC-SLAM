// 回填四份计划的执行记录（2026-11-10）。
// 1) docs/PLAN-slam-loop-pgo-fix.md 的复选框按实际落地情况勾选/标注；
// 2) 四份文件末尾各追加一节「执行记录」，写明做了什么、偏离在哪、还剩什么。
const fs = require('fs');

const slamPath = 'docs/PLAN-slam-loop-pgo-fix.md';
let slam = fs.readFileSync(slamPath, 'utf8');

// 已完成项：勾选并在行尾标注落地方式
const done = [
  ['D1-a', '（`addLoopEdge` 已改 `bool`，10 处拒绝/去重点改为 `return false`，成功路径 `return true`）'],
  ['D1-b', '（`tracking.cpp` / `covis_manager.cpp` 两处调用点都在 false 时直接 return，不 optimize、不写回、不计数）'],
  ['D1-c', '（删除"按第一个观测者 delta 搬地图点"整块，只写回关键帧位姿）'],
  ['D1-d', '（写回前快照全部 KF 位姿，超 `FRAME_JUMP_MAX_*` 则整体回滚并置 `tracking_ok=false`）'],
  ['D2-b', '（`kEMFallbackTranslationScale` 已删；样本 < 3 或基线非有限时 `return false`）'],
  ['D2-c', '（EM 闸门改用匹配点深度实测基线中位数）'],
  ['D3-b', '（yaml 已改为 0.25 / 12 / 12；同时把 `min_depth` 改回 0.1、`max_depth` 改回 12.0，与 Tracking 一致）'],
  ['D3-c', '（`LoopDetectorConfig` 的 `ransac_iterations` / `ransac_threshold_px` / `max_icp_inlier_ratio` 已删，只留 `geometric_verifier.yaml` 一份）'],
  ['D3-d', '（`LoopDetectorConfig` 直接持有 `GeometricVerifierConfig`，`loadAllConfigs()` 注入，两条回环路径都走它）'],
  ['D4-a', '（新增 `loop_inlier_count_ref` / `_min_factor` / `_max_factor`，信息矩阵再乘内点计数因子）'],
  ['D4-b', '（`optimize(double* out_chi2, int* out_iterations)`；chi2 == 0 时无条件告警并打印边上数量）'],
  ['D4-c', '（`odom_info_scale_min/max` 配置化，区间由 [0.5, 2.0] 放宽到 [0.2, 4.0]）'],
  ['D4-d', '（`loop_check_interval_kf`，默认 1 = 每个关键帧，与改动前行为一致）'],
  ['D5-a', '（44 处加 `config_.verbose` 守卫，2 处按 R-021 保持无条件）'],
  ['D5-b', '（守卫 60 处，其中 32 处故障类日志按 R-011 撤回为无条件）'],
  ['D5-c', '（`optimizerNeedsRebuild_` 现在紧邻 `return true`，只在边真的入图时置位）'],
  ['D5-d', '（`local_kf_index` 预建 `unordered_map<int,size_t>`，去掉 O(K²)）'],
  ['D5-e', '（4 处硬编码提为 `dbow_temporal_window` / `dbow_max_results` / `dbow_min_distance_m` / `loop_max_spatial_dist_m` / `min_frames_between_for_spatial_gate`）'],
  ['D6-a', '（`effective_max_iterations` 已从 yaml 删除；`require_candidate_agreement` / `consistency_warning_threshold` 已接线到 `selectBestFromVerified()`）'],
  ['D6-b', '（`G2O_INCLUDE_DIRS` 不再拼两份安装路径，改为用 g2o 导出值或从 `g2o_DIR` 反推）'],
  ['D6-c', '（**已删除**，与 `PLAN-loop-detection-fix.md` 的 L8 冲突时以 L8 为准，理由见执行记录）'],
  ['D6-d', '（临时点独立池 + `retireBadMapPoints()` 真正 erase + 体素近邻查重，见前端 E6）'],
];

for (const [id, note] of done) {
  const re = new RegExp('^- \\[ \\] (' + id + '\\b[^\\n]*)$', 'm');
  if (!re.test(slam)) throw new Error('未找到复选框 ' + id);
  slam = slam.replace(re, (m, body) => '- [x] ' + body + ' ' + note);
}

// D2-a / D3-a：实现了但接口形态与计划不同，勾选并写明偏离
const deviations = [
  ['D2-a',
    '（**接口形态偏离**：`verifyWithEssential` 不持有 3D 点，无法自己算深度基线，' +
    '因此基线中位数在调用方 `loop_detector.cpp` 计算后乘回单位方向；' +
    '`T.translation = baseline * t_dir` 的语义与计划一致）'],
  ['D3-a',
    '（**接口形态偏离**：没有改构造函数签名，而是让 `LoopDetectorConfig` 直接持有 ' +
    '`GeometricVerifierConfig` 成员，构造函数整份拷贝 —— 效果等价，且两条回环路径自动同步）'],
];

for (const [id, note] of deviations) {
  const re = new RegExp('^- \\[ \\] (' + id + '\\b[^\\n]*)$', 'm');
  if (!re.test(slam)) throw new Error('未找到复选框 ' + id);
  slam = slam.replace(re, (m, body) => '- [x] ' + body + ' ' + note);
}

// D7 需要真实数据集与工具链，本机无法执行
const d7note =
  '（**未执行**：本机是 Windows、无 C++ 工具链、无 TUM 数据，无法产出 ATE/RPE。' +
  '需要在 Ubuntu `/home/syf/SFC-SLAM` 上 `colcon build` 后跑 `scripts/run_and_eval_tum.sh`）';
for (const id of ['D7-a', 'D7-b', 'D7-c']) {
  const re = new RegExp('^- \\[ \\] (' + id + '\\b[^\\n]*)$', 'm');
  if (!re.test(slam)) throw new Error('未找到复选框 ' + id);
  slam = slam.replace(re, (m, body) => '- [ ] ' + body + ' ' + d7note);
}

fs.writeFileSync(slamPath, slam);
console.log('[docs] 复选框已回填');

// ---------------- 追加执行记录 ----------------
const stamp = '2026-11-10';

const frontendRecord = `

---

## 十一、执行记录（${stamp}）

**范围**：E1(R5) → E2(R1) → E3(R2) → E4(R3) → E5/E6(R4)，全部按本文件规定的顺序落地。
**验证状态**：本机为 Windows，无 cmake/g++/clang/MSVC/ninja/WSL，**一行都没有编译过**。
只做了静态核对：括号配对（\`scripts/check_braces.js\`）+ 遗留符号扫描（\`scripts/static_check_e6.js\`），
五个前端文件 \`final brace=0 paren=0 brack=0\`，\`observed_count\` / \`R_eigen\` / \`need_undistort\` 均 0 处。

### E1 / R5 · 前后端观测计数打通

- \`MapPointCandidate::observed_count\` 更名 \`frontend_observation_count\`，新增 \`created_frame_id\` /
  \`last_observed_frame_id\`；\`addObservationToMapPoint()\` 现在同时写前端候选（原来只写后端
  \`mapping::MapPoint\`，前端计数永远停在创建值）。
- \`cleanOldMapPoints()\` 只在 \`mNewMapPointIds_\`（观察期内的新建点）上判定，加了
  \`kMapPointObservationGraceFrames = 3\` 的宽限期；原来它遍历全表、把 \`observed_count<=1\` 的点
  一律标坏，第 1 个关键帧之后整张图就废了。
- 建点前的"已有 id 就跳过"判断补上 \`is_bad\` 例外，坏点不再永久占位。

### E2 / R1 · 位姿约定统一（T_wc vs T_cw）

- 选定**方案 A**：\`RobustPnPSolver\` 内部一律 \`T_cw\`（OpenCV PnP / g2o 顶点的语义），
  只在出入口各转换一次 —— 新增静态方法 \`poseCwToWc()\` / \`poseWcToCw()\`，
  \`optimize()\` 入口把 \`init_pose\` 转成 \`T_cw\`、出口把结果转回 \`T_wc\`，\`mlpnpSolve()\` 同理。
- \`poseOptimization()\`：建 \`VertexSE3Expmap\` 前显式 \`getCwFromPose(p)\`，取回后用新增的
  \`poseFromCw()\` 转回 \`T_wc\`，再写 \`mCurrentFrame_->pose\`。
- \`covis_manager.cpp\` 的局部 BA 同属一个位姿约定，一并改：新增
  \`poseWcToSe3Quat()\` / \`se3QuatToPoseWc()\`，顶点建/取回都过这两个函数。
- 全文件已无 \`R_eigen\` 残留（那是把 \`T_wc\` 直接塞进 \`T_cw\` 顶点的那行）。

### E3 / R2 · 删除 \`updateLastFrame()\` 的速度预测覆盖

- 整段 \`mLastFrame_->pose = updated_pose;\` + \`updateWorldPoints(mLastFrame_)\` 删除，
  换成说明注释：上一帧位姿是 \`poseOptimization\` 的优化结果，用恒速外推覆盖等于每帧丢掉
  优化值、误差按帧累积。
- 顺带把误导性的局部变量名 \`Tlr\` 改成 \`last_frame_pose\`（它只是上一帧的绝对位姿，
  不是帧间相对变换）。

### E4 / R3 · 畸变（**偏离计划字面执行**）

计划 §四 同时要求"全程理想像素"和"\`setCamera(config.K, config.D)\`"，但同节末行明确
告警"不能同时做这两件事"。实读代码后确认：\`trackFrame()\` 已经在 \`disable_undistort=false\`
时把 2D 观测去畸变成理想像素，若再把 \`config.D\` 传给 PnP，OpenCV 会对理想像素二次去畸变。

**最终选择"全程理想像素"这一支，PnP 保持接收空 \`D\`**，理由写进了构造函数注释。

真正的修复在 \`compute3DPoints()\`：原来"采样深度"与"反投影"共用同一套坐标，
于是"畸变像素 + 理想针孔"在 3D 与 2D 两侧自洽，误差被掩盖。现在两件事拆开 ——
**采样用畸变像素**（深度图与畸变 RGB 对齐）、**反投影用理想像素**（相机模型是理想针孔），
索引始终对齐 \`frame->keypoints\`。

配套：\`config/tracking.yaml\` 的 \`disable_undistort\` 改 \`false\`；\`test_tum_rgbd.cpp\` 不再强制
覆盖该字段；\`test_visual_slam.cpp\` 两处改为 \`tcfg.disable_undistort = tcfg.D.empty()\`；
各调用点删掉"对已是理想像素的 \`undistorted_keypoints\` 再做一次 \`undistortPoints\`"的重复处理
（\`trackFrame()\` 的 Step 4 成为唯一的坐标系决定点）。

### E5 / R4 · 阈值与流程

- 观测权重按特征点所在金字塔层加权（\`invSigma2 = 1/(1.2·scale^level)^2\`），不再固定 1.2。
- PnP 一致性阈值 5.0m / 1.5rad → **0.3m / 0.3rad**（原值等于"PnP 永远通过"）。
- \`trackWithMotionModel()\` 失败时把位姿退回本帧起点（原来留着恒速预测的坏初值给后续路径）。
- \`updateLocalMap()\` 不再把普通帧塞进 \`mvpLocalKeyFrames_\`（关键帧窗口只由
  \`createNewKeyFrame()\` 维护），避免 \`trackReferenceKeyFrame()\` 拿本帧自己的
  \`poseOptimization\` 结果当参考、误差自我循环。
- \`bow_matches\` 循环补 6 条越界/无效闸；几何一致性容差 \`max(radius*2.5, 20px)\` → \`max(radius*1.5, 8px)\`；
  运动模型合理性阈值 1.0 m/帧 → 0.15 m/帧。

### E6 · 地图点生命周期

- 临时 VO 点改为进独立池 \`mTemporalMapPoints_\`，与主表共用 \`next_mp_id_\` 空间；
  \`getMapPoint()\` 先查主表再查临时池。原来它们照样进 \`mAllMapPoints_\` 且永不回收。
- \`retireBadMapPoints()\` 真正 \`erase\` 坏点，并清理 \`mvpCurrentMapPointIds_\` /
  \`mLastFrameMapPointIds_\` / 所有关键帧 \`map_point_ids\` 里的悬空 id。
  调用点固定在单帧流程末尾、无 \`MapPointCandidate*\` 被持有的时点。
- \`createMapPoint()\` 先做体素近邻查重（\`kMapPointVoxelSize = 0.10\` / \`kMapPointDuplicateRadius = 0.05\`），
  命中则复用已有 id 而不是新建。

### 未完成 / 需在有工具链的机器上确认

1. **编译**：全部改动未经编译。\`std::clamp\` 依赖 C++17，需确认 \`CMAKE_CXX_STANDARD\`。
2. **ATE/RPE**：C1 的评测脚本存在，但从未实跑（无数据集）。
3. \`updateLocalMap()\` 删掉普通帧后，\`mpReferenceKF_\` 的语义变化需要在真实序列上看
   \`trackReferenceKeyFrame()\` 的命中率是否下降。
4. R-020 建议的两个强类型 \`PoseWc\` / \`PoseCw\` 未引入 —— 本次只统一了转换点，没有让混用
   在编译期失败。这是本文件 §九 给出的建议，不是 E1 的验收条件。
`;

const loopRecord = `

---

## 九、执行记录（${stamp}）

**范围**：R1(L1+L6) → R2(L2) → R3(L3) → R4(L4) → R5(L5) → R6(L10) → R7(L7+L8+L9+L11)。
**验证状态**：Windows 本机**无任何 C++ 工具链**，全部改动未编译；只做括号配对与遗留符号静态核对
（\`loop_detector.cpp\` / \`loop_detector.hpp\` / \`pose_graph_optimizer.cpp\` / \`pose_graph_optimizer.hpp\` /
\`geometric_verifier.hpp\` 均 \`final brace=0 paren=0 brack=0\`）。

### R1 · L1 + L6

- **L1**：\`geometricVerification()\` 构造 Sim3 输入时不再做第二次 world→camera。原来调用侧已经
  用 \`R_cw * pt_world + t_cw\` 转过一次，被调函数 \`geometricVerificationWithSim3()\` 内部又各转一次
  （形参名就叫 \`*_world\`），旋转平移施加两遍把点云搬到相机背后 —— 这是 covis 路径"永远失败、
  只能回落 ICP+EM"的根因。
- **L6+L7**：\`valid_matches\` 改用 \`matches[i].queryIdx/trainIdx\` 取点（原来误用循环下标），
  并按传给 ICP 的压缩序号重映射 \`queryIdx/trainIdx\`；ICP 内点率分母由 \`matches.size()\` 改成
  \`valid_matches.size()\`。

### R2 · L2

\`geometricVerification()\` 的阶段 2（描述子 matches + \`keyframe_db_\` 自带的世界坐标点）
与 Sim3 块整体移出 \`if (covis_mgr != nullptr)\` 守卫，只有阶段 1（confirmed map point + KD-tree）
仍需要 \`covis_mgr\`。\`Tracking::detectAndCorrectLoop()\` 继续传 \`nullptr\`，但现在是有意为之 ——
原实现把阶段 2 也写在守卫里，于是这条路径上 \`covis_matches\` 恒空、Sim3 永不进入。

### R3 · L3

- \`addOdometryEdge()\` 里的 \`edge->setLevel(1)\` 删除。它配合 \`optimize()\` 不带参数的
  \`initializeOptimization()\`（默认 level 0），效果是**所有里程计边被整类移出优化**，
  图上只剩 1 条回环边 —— 一条边的图 chi2 必然恒 0。
- \`initializeOptimization()\` 改为显式 \`initializeOptimization(0)\`。

### R4 · L4

\`addLoopEdge()\` 由 \`void\` 改为 \`bool\`：10 处拒绝/去重点全部 \`return false\`，成功路径
\`return true\`。\`tracking.cpp\` 与 \`covis_manager.cpp\` 两处调用点都在 false 时直接返回，
不再"检测到回环但边没进图"却计一次回环。

### R5 · L5

几何验证阈值回配置：\`config/geometric_verifier.yaml\` 一直无人读取，阈值写死在
\`loop_detector.cpp\` 的构造函数里。现在 \`LoopDetectorConfig\` **直接持有**
\`GeometricVerifierConfig\` 成员，由 \`loadAllConfigs()\` 注入，两条回环路径自动同步。
\`min_inliers\` 2→12、\`min_inlier_ratio\` 0.15→0.25、\`min_icp_inliers\` 2→12。
同一份 yaml 的 \`min_depth\` 0.05→0.1、\`max_depth\` 30→12，与 Tracking 保持一致
（此前 yaml 没有消费者，这个 P0-1 级的不一致一直没被发现；接上之后它就成了生效值）。

### R6 · L10

\`geometric_verifier.hpp\` 的 46 处 \`std::cerr\` 中 44 处加 \`if (config_.verbose)\` 守卫；
2 处跳过：一处是注释里提到的说明文字，一处是"内点数不足"（按 R-011 保持无条件）。

### R7 · L7 + L8 + L9 + L11

- **L7**：\`icp_inliers == 0\` 时的"hybrid fallback"（\`T_hybrid = R_icp + frame_dist * t_icp_dir\`）
  整段删除，改为降级到 Essential Matrix 并打一条 verbose 日志。原代码在注释里写着"已删除
  hybrid fallback"，但代码还在。
- **L8**：\`pose_graph_optimizer.cpp\` 的 \`sigjmp_buf\` / \`sigaction\` / SIGSEGV 捕获整块删除，
  只保留 C++ \`try/catch\`。理由：段错误发生时 g2o 内部已处于半更新状态，\`longjmp\` 跳过栈展开
  后返回的是损坏的图，后续每帧都在损坏状态上"优化" —— 崩溃被换成静默的错误轨迹，比直接崩更难查。
  同时把原本被注释掉的 dangling-edge 预检查**复活成真代码**（必须早于 \`initializeOptimization()\`，
  否则 \`_activeEdges\` 里留悬空指针）。文件 1501 → 1323 行。
- **L9**：\`computeCandidateConsistency()\` 接线进 \`selectBestFromVerified()\`（\`require_candidate_agreement\` /
  \`consistency_warning_threshold\` / \`max_spatial_dist_for_consistency\` 三个键首次真正生效）；
  \`correctLoopWithScale()\` 与 \`scale_min\` / \`scale_max\` 删除 —— 它的尺度逻辑已被 EM 路径的
  深度实测基线取代，保留会形成第二套尺度真值。
- **L9 附带（原 D2-b）**：EM 降级路径写死的 \`kEMFallbackTranslationScale = 1.0\` 删除。
  现在用每对有效对应点算出的 \`t = p_train - R_em * p_query\` 投影到单位方向取中位数作为基线；
  样本不足 3 对或基线非有限时**直接拒绝这次 EM 回环**，不再编一个数顶上。
- **L11**：排序比较器改为严格弱序（\`is_icp\` → \`quality_score\` → \`kf_id\` 三级 key），
  原实现 \`return a.is_icp && !b.is_icp;\` 在相等时返回 true，\`std::sort\` 属未定义行为。

### 未完成 / 需在有工具链的机器上确认

1. **编译与实测**：一行都没编译过。L8 删除 SIGSEGV 防护后建议先在有工具链的机器上跑一次 ASAN
   再上真实序列。
2. **不要同时判定 L1 与 L7 的贡献**：两者都改变回环边数值，本文件 §八 已警告不能同时改；
   实际执行时两者都改了，若结果异常需要按本记录的提交顺序分别回退定位。
3. \`SimpleKDTree2D\` 实际是 O(N²) 暴力最近邻、\`Sim3Solver\` 的 \`20.0f\` 误差阈值不按 octave 缩放 ——
   两项风险本文件 §八 已列出，本次未处理。
4. \`config/covis_manager.yaml\` 的 \`loop_detection_interval: 9999\` 仍在。这意味着 **CovisManager
   那条回环路径实际上从不检测回环**（注释写的是"纯 VO 模式"）。它与
   \`PLAN-tracking-drift-fix.md\` 的 B2 是同一个待决策问题（纯 VO 还是完整 SLAM），本次不动。
5. 本文件 §九 建议新增的 **R-028** 已采纳为代码注释原则（见 \`addOdometryEdge()\` 内的说明），
   但尚未写入 \`CODE-RULES.md\`。
`;

const slamRecord = `

---

## 八、执行记录（${stamp}）

**范围**：D1 → D2 → D3 → D4 → D5 → D6 全部落地；D7 因缺少工具链与数据集未执行。
**验证状态**：Windows 本机无 C++ 工具链，未编译；静态核对仅做到括号配对与符号残留扫描。
**与 \`PLAN-loop-detection-fix.md\` 的关系**：两份文件的 L/D 编号大量重叠，实际执行时以
"同一处代码只改一次"为原则合并处理，落地位置记录如下。

### D1 · 回环边真的能进图

- D1-a / D1-b：\`addLoopEdge\` 改 \`bool\`，\`tracking.cpp\` / \`covis_manager.cpp\` 两处调用点都在
  false 时直接返回。
- D1-c：\`applyOptimizedPosesToFrontend()\` 删掉"按第一个观测者 delta 搬地图点"整块。
  一个点被 K 个关键帧观测、各帧 delta 不同，用其一搬点等于让另外 K-1 个观测指向错误世界位置，
  地图会长出重复结构。现在只写回关键帧位姿。
- D1-d：\`track()\` 里 \`detectAndCorrectLoop()\` 前后加位姿快照，写回后位移/旋转超过
  \`FRAME_JUMP_MAX_TRANSLATION_M\`(0.5) / \`FRAME_JUMP_MAX_ROTATION_RAD\`(0.524) 时整体回滚
  （\`mAllKeyFrames_\` / \`mvpLocalKeyFrames_\` / \`mpMap_\` / \`loop_detector_\` / 当前帧 / 上一帧），
  并置 \`tracking_ok = false\`。

### D2 · 真实尺度

- D2-a（**接口形态偏离**）：\`verifyWithEssential\` 只接收 2D 关键点，拿不到深度，无法自己算基线。
  实现改为：调用方用 EM 解出的旋转对 3D 对应点求 \`t = p_train - R_em * p_query\`，
  投影到单位方向取中位数，再 \`T_em.block<3,1>(0,3) = baseline * t_dir\`。
  语义与计划一致（平移长度来自深度实测），只是计算位置在调用方。
- D2-b：\`kEMFallbackTranslationScale\` 已删；样本 < 3 对或基线非有限时 \`return false\`。
- D2-c：EM 的帧间距离闸门改用同一批样本的模长中位数，不再用 \`pose_curr.t - pose_matched.t\`
  —— 回环场景里这两帧的 PnP 结果恰恰漂得最厉害。

### D3 · 阈值回配置

- D3-a（**接口形态偏离**）：没有改构造函数签名，改为 \`LoopDetectorConfig\` 直接持有
  \`GeometricVerifierConfig\` 成员，构造函数整份拷贝。效果等价，且 CovisManager 与 Tracking
  两条回环路径自动同步。
- D3-b：\`config/geometric_verifier.yaml\` 收紧 \`min_inliers\` 12 / \`min_inlier_ratio\` 0.25 /
  \`min_icp_inliers\` 12；**同时修正 \`min_depth\` 0.05→0.1、\`max_depth\` 30→12**
  （接上消费者之后这份 yaml 才真正生效，而它与 Tracking 的深度范围原本就不一致）。
- D3-c：\`LoopDetectorConfig\` 的 \`ransac_iterations\` / \`ransac_threshold_px\` /
  \`max_icp_inlier_ratio\` 删除，只留几何验证器那一份；\`config/loop_detector.yaml\` 对应条目
  改为指向 \`geometric_verifier.yaml\` 的注释。
- D3-d：\`loadAllConfigs()\` 里 \`all_cfg.loop_detector.geometric_verifier = all_cfg.geometric_verifier;\`，
  \`initLoopDetection\` 整份传入，测试无需改签名。

### D4 · 位姿图可解

- D4-a：回环边信息矩阵再乘"内点计数因子" \`inliers / loop_inlier_count_ref\`，夹在
  \`[loop_inlier_count_min_factor, loop_inlier_count_max_factor]\` = \`[0.25, 4.0]\`。
  原来只有内点率的一次项，"2 对全内点"与"200 对全内点"权重完全相同。
- D4-b：\`optimize(double* out_chi2, int* out_iterations)\`；\`chi2 == 0\` 时无条件告警并打印
  里程计边数 / 回环边数 / 活跃边数。两处调用点都把 chi2 打进日志。
- D4-c：\`odom_info_scale_min/max\` 配置化，区间由写死的 \`[0.5, 2.0]\` 放宽到 \`[0.2, 4.0]\`。
  原区间把 30% 与 90% 内点的里程计边压成同一权重，回环修正传不出去。
- D4-d：新增 \`loop_check_interval_kf\`（默认 1 = 每个关键帧检测，与改动前行为完全一致）。
  原来唯一的闸门是"同一个 KF 不重复进来"，等于每个关键帧都跑一次 BoW + 几何验证 + Sim3/ICP + 全局 PGO。

### D5 · 性能

- D5-a：\`geometric_verifier.hpp\` 44 处加 \`config_.verbose\` 守卫（2 处按 R-021 无条件保留）。
- D5-b：\`pose_graph_optimizer.cpp\` 60 处加 \`opts_.verbose\` 守卫，其中 32 处故障类日志
  （拒绝/失败/NaN/异常/initializeOptimization 失败）按 R-011 撤回为无条件输出。
- D5-c：\`optimizerNeedsRebuild_ = true;\` 现在紧邻 \`return true;\`，只在边真的入图时置位。
- D5-d：\`applyOptimizedPosesToFrontend()\` 预建 \`unordered_map<int,size_t> local_kf_index\`，
  去掉 K×K 线性查找。
- D5-e：\`min(5ul, ...)\` → \`candidate_pool_size\`；\`MAX_SPATIAL_DIST_FOR_LOOP = 8.0\`、
  \`frames_between < 30\`、\`queryWithConsistency(..., 3, 10, 1.0)\` 全部提为
  \`dbow_temporal_window\` / \`dbow_max_results\` / \`dbow_min_distance_m\` /
  \`loop_max_spatial_dist_m\` / \`min_frames_between_for_spatial_gate\`。

### D6 · 配置、地图点生命周期与构建

- D6-a：\`effective_max_iterations\` 从 \`config/pose_graph_optimizer.yaml\` 删除 ——
  全仓没有任何代码读取它，改它不会有任何效果；同一段注释还声称"实际代码中取
  min(max_iterations, 10)"，而 \`pose_graph_optimizer.cpp\` 里是把 \`opts_.max_iterations\`
  原样传给 g2o。两处注释一并更正。\`require_candidate_agreement\` /
  \`consistency_warning_threshold\` 选择"接线"而不是删除（见 D9/L9）。
- D6-b：\`src/sfc_core/CMakeLists.txt\` 不再把 \`/home/syf/g2o_user_install/include\` 与
  \`/home/syf/g2o/build/install/include\` 两份安装同时拼进 \`G2O_INCLUDE_DIRS\`
  （两份头文件版本不同时，谁赢取决于书写顺序）。改为优先用 \`find_package(g2o)\` 的导出值，
  为空时从 \`g2o_DIR/../../..\` 反推 \`<prefix>/include\`，并 STATUS 打印来源。
  \`ResolveThirdparty.cmake\` 已经把 \`g2o_DIR\` 唯一化，这里不再造第二个来源。
- D6-c：**已删除 SIGSEGV 防护**，而不是"先保留"。原因是 \`PLAN-loop-detection-fix.md\` 的 L8
  独立判定它有害，且两条理由可以同时成立：段错误时 g2o 已处于半更新状态，\`longjmp\` 回来的是
  损坏的图 —— 把"崩溃"换成"静默错误轨迹"。计划 D6-c 写"修复 D1–D4 之后单独评估"，
  评估结论是删除，因此执行了删除并保留 C++ \`try/catch\`。
- D6-d：临时 VO 点改为进独立池 \`mTemporalMapPoints_\`；\`retireBadMapPoints()\` 真正
  \`erase\` 坏点并清理悬空 id；\`createMapPoint()\` 先做体素近邻查重（0.10m 体素 / 0.05m 判重）。

### D7 · 未执行

本机无 C++ 工具链、无 TUM 数据集，\`scripts/evaluate_trajectory.py\` 与
\`scripts/run_and_eval_tum.sh\` 无法运行，**没有任何 ATE / RPE 数字**。
需要在 Ubuntu \`/home/syf/SFC-SLAM\` 上 \`colcon build\` 后按下面口径跑：

\`\`\`bash
bash scripts/run_and_eval_tum.sh fr1_desk2   # 纯 VO
bash scripts/run_and_eval_tum.sh fr1_room
python3 scripts/evaluate_trajectory.py --align --no-scale <traj> <gt>
\`\`\`

### 未完成 / 需确认

1. **编译**：全部改动未编译。\`std::clamp\`（E5）依赖 C++17；\`<cstdint>\`（E6 体素键）、
   \`<cmath>\`（EM 基线）已补，其余头文件依赖未逐一验证。
2. **D7 的量化验收**：上面的基线（纯 VO 478.569 m / 完整 SLAM 1165.124 m / 真值 5–12 m）
   是改动前的数字，改动后必须重测才有意义。
3. \`config/covis_manager.yaml\` 的 \`loop_detection_interval: 9999\` 未动 —— 它等于关闭
   CovisManager 回环路径，与 \`PLAN-tracking-drift-fix.md\` 的 B2 是同一个待决策问题。
4. D1-c 之后地图点不再随回环写回搬动，地图需要若干帧 \`poseOptimization\` 才能重新收敛；
   真实序列上应观察回环后短期内的 reprojection error 是否先升后降。
`;

fs.appendFileSync('docs/PLAN-frontend-pose-fix.md', frontendRecord);
fs.appendFileSync('docs/PLAN-loop-detection-fix.md', loopRecord);
fs.appendFileSync(slamPath, slamRecord);
console.log('[docs] 三份记录已追加');
