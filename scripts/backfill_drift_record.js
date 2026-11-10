// 给 docs/PLAN-tracking-drift-fix.md 追加执行记录（2026-11-10）。
const fs = require('fs');

const record = `

---

## 七、执行记录（2026-11-10）

**状态**：A1–A4、B1、B3、C1 的代码在此之前已在位（本文件已勾选）。本次执行另外两份
计划（\`PLAN-frontend-pose-fix.md\` / \`PLAN-loop-detection-pgo-fix.md\`）时，有若干改动
直接落在这个文件的根因上，一并记录，避免以后把同一处修复当成两件事。

**验证状态**：Windows 本机无 C++ 工具链、无 TUM 数据集，**未编译、未跑评测**。
C1 的脚本存在但从未实跑，因此本文件里没有任何 ATE/RPE 数字。

### 本次与 P0-1（poseOptimization 外点剔除）相关的改动

- \`poseOptimization()\` 的 g2o 顶点现在显式从 \`T_wc\` 转成 \`T_cw\`（\`getCwFromPose\`），
  取回后再转回 \`T_wc\`（\`poseFromCw\`）。在此之前，\`Pose\` 被当作 \`T_wc\` 使用，
  却原样喂给按 \`T_cw\` 解算的 \`VertexSE3Expmap\` + \`EdgeSE3ProjectXYZ\` ——
  **A1 恢复的外点剔除标准，是建立在一组位姿约定错误的残差上的**。
  两者现在一致，A1 的 \`CHI2_THRESHOLD_2DOF_95 = 5.99\` 才真正作用在正确的误差上。
- \`RobustPnPSolver\` 的对外出口统一返回 \`T_wc\`（内部一律 \`T_cw\`，只在出入口各转一次）。
- 观测权重改为按金字塔层加权（\`invSigma2 = 1/(1.2·scale^level)^2\`），原来固定 \`1.2\`，
  等于把高层模糊特征和零层清晰特征当成同等可信。

### 本次与 P1-5（纯 VO 无兜底）相关的改动

- \`config/covis_manager.yaml:178\` 的 \`loop_detection_interval: 9999\` **仍未动**，
  即 CovisManager 那条回环路径依然实际关闭。B2 仍是待决策项（纯 VO 还是完整 SLAM），
  本次不擅自改变产品行为。
- 但原来"即使开了回环也不会生效"的两处 bug 已修：\`addOdometryEdge()\` 的
  \`setLevel(1)\`（把全部里程计边移出优化）+ \`initializeOptimization()\` 默认 level 0，
  合起来让位姿图只剩 1 条回环边、chi2 恒 0；以及 \`geometricVerification()\` 的
  Sim3 输入被两次 world→camera 变换（点云被搬到相机背后，covis 路径永远失败）。
  也就是说：**B2 一旦决定走完整 SLAM，兜底路径这次是真的能工作了**。

### 本次与 P2-6（帧间跳变）相关的改动

- B3 的帧间跳变保护扩展到回环写回：\`track()\` 里 \`detectAndCorrectLoop()\` 前后快照
  全部关键帧位姿，写回后位移 > 0.5 m 或旋转 > 0.524 rad 则整体回滚并置
  \`tracking_ok = false\`。原来全局 PGO 的写回没有任何幅度检查，
  而位姿图这次开始真的会移动顶点。
- \`updateMotionModel()\` 的合理性阈值 1.0 m/帧 / 45° → **0.15 m/帧 / 30°**
  （室内 30 Hz 手持每帧真实位移 3–7 cm，原值等于对恒速模型无约束）。
- \`updateLastFrame()\` 删除"用恒速外推覆盖上一帧已优化位姿"的整块 —— 那是漂移的直接来源之一：
  每帧丢一次优化结果，误差按帧累积。
- \`trackWithMotionModel()\` 失败时把位姿退回本帧起点，不再把恒速预测的坏初值留给
  \`trackReferenceKeyFrame()\` / 重定位。
- 几何一致性容差 \`max(radius*2.5, 20px)\` → \`max(radius*1.5, 8px)\`；
  PnP 与上一帧的一致性阈值 5.0 m / 1.5 rad → 0.3 m / 0.3 rad（原值等于"PnP 永远通过"）。

### 仍未做

1. **B2 待决策**（\`loop_detection_interval: 9999\` 是否保留）——本次刻意不动。
2. **C2 未做**：没有 ATE/RPE，无法定位残留漂移来源。
3. **编译**：所有改动未经编译。\`std::clamp\`（观测权重）依赖 C++17，
   需确认 \`src/sfc_core/CMakeLists.txt\` 的 \`CMAKE_CXX_STANDARD\`。
4. 阶段 A 的回滚清单仍为空 —— 项目不是 git 仓库，回滚依赖备份副本，
   建议在有工具链的机器上先把当前状态打一个 tag 或整目录快照再编译。
`;

fs.appendFileSync('docs/PLAN-tracking-drift-fix.md', record);
console.log('[docs] PLAN-tracking-drift-fix.md 记录已追加');
