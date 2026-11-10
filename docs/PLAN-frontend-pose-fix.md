# 前端位姿估计修复计划（纯 VO 每帧位姿翻转）

> 状态：待执行。本轮只做诊断，未改任何代码。
> 上一份 `docs/PLAN-slam-loop-pgo-fix.md` 处理的是回环/PGO 层；用户已指出「纯 VO 数据都很差，回环肯定回不上来」，本文件针对**前端位姿估计本身**。

---

## 一、结论

**系统里存在两套互不相容的位姿约定，而且两边都在用。**

- 文档与全部坐标助手函数按 **T_wc（camera→world）** 实现；
- PnP（OpenCV）与 g2o 位姿顶点按 **T_cw（world→camera）** 工作；
- 两条路径产出的位姿被直接互相赋值，**没有任何一处做求逆或转置**。

小角度下 `T_cw ≈ [I | −t_wc]`，所以这个错误的主要表现就是：**旋转项变化不大，平移整体取反**。诊断日志正是这个形态 —— `[poseOpt-DBG]` 里 `rot_diff` 一直只有 0.01–0.23 rad，而 `t` 在 f=2/3/4 连续三次整体翻号（`before=(+0.0355,-0.0235,-0.2370)` → `after=(-0.0622,+0.0221,+0.1590)`），`delta_t` 依次 0.068 → 0.3914 → 0.8527 → 1.0432 m 单调递增。这是 640 帧累积 478 m 的直接来源。

> 注：早期把这归因于 `R_wc_det=1.0000` 是错的 —— 那是 `q.toRotationMatrix().determinant()`，任何合法旋转都等于 1，与坐标约定无关，不构成证据。f=0 也不能用来判别，因为位姿接近单位阵时两种约定数值几乎相同。判别力来自 f≥2 的符号翻转。

按影响排序的三条根因：

| # | 根因 | 影响 | 代码位置 |
|---|---|---|---|
| R1 | 位姿约定混用（T_cw 被当 T_wc） | 每帧平移取反，误差线性累积 | `tracking.cpp:61-80`、`:2438-2439`、`:1195-1196`；`robust_pnp_solver.cpp:50-66`、`:734-736` |
| R2 | `updateLastFrame()` 用速度预测覆盖已优化位姿 | 上一帧位姿每帧被推歪 0.13–0.18 m，正反馈 | `tracking.cpp:943-959` |
| R3 | 畸变模型与理想针孔不一致 | 边缘 3D 点方向偏斜，且被"自洽"掩盖 | `tracking.cpp:337-384`、`:257`、`:2350` |
| **R5** | **前后端两套观测计数互不相通 → 第一个关键帧就清空整张前端地图，且死点永久阻止重建** | **建图能力从第 1 个 KF 起归零；`poseOptimization` 拿不到对应点 → 反复 `return 0` → LOST → reloc** | `tracking.cpp:324-332`、`:3218-3231`、`:3062-3065`、`:979-982`、`:2785`、`:288` |
| R6 | 判别证据来源需注意版本 | `src/diag_log2.txt` 是阶段 A 修复**之前**的日志（仍有 `th=10000.0000`，且 `TRKF-FILTER`/`PnP-RANSAC` 在当前代码里 grep 为 0）。定性结论仍成立，但行号/阈值一律以当前代码为准 | `src/diag_log2.txt` vs `tracking.cpp` |

---

## 二、R1：位姿约定混用（P0，其他问题的前提）

### 2.1 证据一：助手函数按 T_wc 实现

`src/sfc_core/src/frontend/tracking.cpp:61-80`

```cpp
// 从 Pose (T_wc) 获取 world→camera 的变换 (T_cw = T_wc^{-1})
inline std::pair<Eigen::Matrix3d, Eigen::Vector3d> getCwFromPose(const Pose& pose) {
    Eigen::Matrix3d R_cw = pose.q.toRotationMatrix().transpose();
    Eigen::Vector3d t_cw = -R_cw * pose.t;
    return {R_cw, t_cw};
}
inline Eigen::Vector3d camToWorld(const Eigen::Vector3d& pt_cam, const Pose& pose) {
    return pose.q.toRotationMatrix() * pt_cam + pose.t;   // T_wc
}
inline Eigen::Vector3d worldToCam(const Eigen::Vector3d& pt_world, const Pose& pose) {
    auto [R_cw, t_cw] = getCwFromPose(pose);
    return R_cw * pt_world + t_cw;                        // T_wc^{-1}
}
```

`Pose` 在这套语义里就是 **T_wc**。全仓 `camToWorldPoint` / `worldToCam` / `projectWorldToCam` / `checkBidirectionalConsistency` 都依赖这个前提。

### 2.2 证据二：PnP 返回的是 T_cw

`src/sfc_core/src/frontend/robust_pnp_solver.cpp:50-66`

```cpp
double reprojError(const cv::Point3f& pt3d, const cv::Point2f& pt2d,
                   const Pose& pose, const Eigen::Matrix3d& K) {
    Eigen::Vector3d pw(pt3d.x, pt3d.y, pt3d.z);
    Eigen::Matrix3d R = pose.q.toRotationMatrix();
    Eigen::Vector3d pc = R * pw + pose.t;      // world → camera，pose 是 T_cw
    ...
}
```

`:585-593` 直接调用 `cv::solvePnPRansac(pts3d, pts2d, K, D, rvec, tvec, ...)`；OpenCV 的约定是 `p_cam = R * p_world + t`，即 **T_cw**。
`:734-736` 把结果原样包装：

```cpp
Eigen::Quaterniond q(R_eigen);
q.normalize();
best_pose = Pose(q, t_eigen);      // 没有任何 .inverse()
```

`:1073` BA 路径同理：`current_pose = Pose(quat(T_opt.rotation()), T_opt.translation());`
`:1470` 最终 `result.pose = final_pose;` —— 仍是 T_cw。

**全文件 grep `.inverse()` 零命中**，只有对打印向量用 `.transpose()`。也就是说 `RobustPnPSolver` 输出的 `Pose` 从始至终是 T_cw。

### 2.3 证据三：g2o 位姿顶点按 T_cw 解释

`src/sfc_core/src/frontend/tracking.cpp:2436-2439`

```cpp
// 添加单个位姿顶点（当前帧的 Tcw）      ← 注释自己写了 Tcw
g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
Eigen::Matrix3d R_eigen = p.q.toRotationMatrix();
vSE3->setEstimate(g2o::SE3Quat(R_eigen, p.t));
```

`g2o::EdgeSE3ProjectXYZOnlyPose`（`src/sfc_core/include/sfc_core/mapping/optimizer.hpp:35-38` 只是纯别名，没有自定义投影）的残差是

```
_error = obs - cam_project( v->estimate().map(Xw) )
```

`SE3Quat::map(Xw) = R*Xw + t`，所以**顶点必须装 T_cw**。自写的 `EdgeRobustReproject` 是同一套语义（`src/sfc_core/include/sfc_core/frontend/robust_pnp_solver.hpp:50-55`）：

```cpp
g2o::SE3Quat T = v_se3->estimate();
Eigen::Vector3d xyz_cam = T.map(_pt3d);      // T 必须是 T_cw
```

### 2.4 证据四：赋值链上没有任何转换

| 行 | 代码 | 语义 |
|---|---|---|
| `tracking.cpp:1195-1196` | `mCurrentFrame_->pose = robust_pnp_solver_->optimize(pts3d, pts2d, ...)` | T_cw 写进 T_wc 槽 |
| `tracking.cpp:2438-2439` | 把 `mCurrentFrame_->pose` 送进 g2o 顶点 | 恰好是 T_cw，能对上 |
| `tracking.cpp:2626-2629` | 从顶点取回 `new_pose` | T_cw |
| `tracking.cpp:2758` | `mCurrentFrame_->pose = new_pose;` | 又存成"T_wc" |
| `tracking.cpp:1270` | `updateWorldPoints(mCurrentFrame_)` | 用 `camToWorldPoint(pts_3d, pose)` —— 把 T_cw 当 T_wc |
| `tracking.cpp:1137` | `camToWorldPoint(p3d_cam, ref_kf.world_pose)` | 同上，参考帧位姿也被当 T_wc |
| `tracking.cpp:2383` | `Eigen::Vector3d pt_cam = worldToCam(Xw, p);` | 诊断用 T_wc 解释 T_cw |

### 2.5 这解释了所有观测现象

| 现象 | 解释 |
|---|---|
| `R_wc_det=1.0000` 每帧都成立 | 旋转矩阵行列式与约定无关；且小角度下 `R_cw ≈ R_wc` |
| `t` 逐帧翻号（f=2 `−0.2370 → +0.1590`、f=3 `+0.4402 → −0.3822`） | `T_cw` 的平移是 `−R_wc^T · t_wc`，小角度下就是 `−t_wc` |
| `init_reproj_err_avg` f=0 就 10 px、f=1 已经 64 px | `:2383` 用错约定的 `worldToCam(Xw, p)` 算出来的**假误差**，不是真实重投影残差 |
| `delta_t` 单调递增 0.068 → 0.39 → 0.85 → 1.04 m | 每帧把上一个错误位姿当基准，误差按帧累积 |
| 640 帧 478 m（≈0.75 m/帧） | 逐帧累积的直接积分 |
| `inliers` 常年 800–1000 | 描述子匹配本身没问题，错的是几何解释；纯 2D 匹配数照旧很高 |
| `[poseOpt f=4] REJECTED: t_diff=1.0057` 拦住了翻转 | 1.0 m 阈值只是偶然触发；到 f=610+ 之后 5 m 阈值全程放行 |

### 2.6 修法（三选一，必须选且只能选一个）

**方案 A（推荐）：全系统统一到 T_wc，在 PnP 出口做一次求逆。**

- `RobustPnPSolver::optimize` 返回前把 T_cw 转成 T_wc 再返回：
  ```cpp
  // 内部一律 T_cw（OpenCV/g2o 约定）；对外统一 T_wc（本系统约定）。
  // 转换只在这一处发生，其余位置不得再做逆变换。
  Pose toWorldPose(const Pose& t_cw) {
      Eigen::Matrix3d R_cw = t_cw.q.toRotationMatrix();
      Pose out;
      out.q = Eigen::Quaterniond(R_cw.transpose());
      out.t = -R_cw.transpose() * t_cw.t;
      return out;
  }
  ```
- **同时**把 `RobustPnPSolver` 内部改成显式命名（`Pose t_cw` 而不是 `Pose pose`），并在 `reprojError` 的注释里写明「入参是 T_cw」。
- `poseOptimization` 的顶点仍传 T_cw，但要在建边前显式 `getCwFromPose(p)` 得到 T_cw，而不是靠"碰巧 p 就是 T_cw"：
  ```cpp
  auto [R_cw, t_cw] = getCwFromPose(p);          // 显式转换，不依赖上游巧合
  vSE3->setEstimate(g2o::SE3Quat(R_cw, t_cw));
  ...
  // 取回后转回 T_wc
  mCurrentFrame_->pose = fromCw(SE3Quat(vSE3->estimate()));
  ```
- 诊断 `:2383` 的 `worldToCam(Xw, p)` 保持不变（p 已是 T_wc，正确）。

**方案 B：全系统统一到 T_cw（对齐 ORB-SLAM3）。**
改动面更大（所有 `camToWorld`/`worldToCam` 调用点、`pts_3d_world`、`KeyFrameCacheEntry::world_pose`、PGO、回环全要翻），但语义与 ORB-SLAM3 一致，后续维护成本低。代价是这一轮要改 100+ 处。

**方案 C（否决）：在 g2o 边里自定义投影函数做转换。**
把约定错误藏进边内部，下次换优化器还得再错一遍。**不采用。**

> 决策建议：先做 A 拿回可测量的正确性，B 留作可读性整改阶段统一重构。

---

## 三、R2：`updateLastFrame()` 用速度预测覆盖已优化位姿（P0）

### 代码（`src/sfc_core/src/frontend/tracking.cpp:943-959`）

```cpp
if (mLastFrame_ && mbVelocityValid_) {
    Eigen::Matrix3d R_delta = mVelocity_.q.toRotationMatrix();
    Eigen::Vector3d t_delta = mVelocity_.t;
    Pose updated_pose;
    updated_pose.q = mVelocity_.q * mLastFrame_->pose.q;
    normalizeQuaternion(updated_pose.q);
    updated_pose.t = R_delta * mLastFrame_->pose.t + t_delta;
    mLastFrame_->pose = updated_pose;   // ← 覆盖已经优化过的位姿
    updateWorldPoints(mLastFrame_);     // ← 世界点按预测位姿全部重算
}
```

### 日志证据

| 帧 | `[DBG n] pose=`（真实结果） | 下一帧 `[SearchByProjection] last_pose=` | 凭空位移 |
|---|---|---|---|
| f=1 → f=2 | `(-0.0240, -0.0213, 0.0178)` | `(0.006, -0.022, -0.109)` | 0.127 m |
| f=2 → f=3 | `(-0.0213, 0.0009, 0.0740)` | `(-0.052, 0.021, 0.257)` | 0.183 m |

### 危害链

1. `mLastFrame_->pose` 与 `mAllKeyFrames_[id].world_pose` 从此不一致（同一个关键帧两套坐标）。
2. `pts_3d_world` 被按预测位姿重算 → 地图点世界坐标整体漂移，而 `mAllMapPoints_` 里的点不跟着变。
3. 下一帧 `searchByProjection` 的搜索中心就是按被污染的位置。
4. 速度越错 → 覆盖越离谱 → 下一帧初值越错 → 正反馈。

### 修法

- **删除这段位姿覆盖**。`updateLastFrame()` 只保留后半段「为上一帧没有地图点的深度特征建临时点」的功能。
- 需要"用参考关键帧修正上一帧位姿"时，照 ORB-SLAM3 的做法用 `Tlr`（参考帧→当前帧的相对位姿）只作用于**临时点池**，不改写 `mLastFrame_->pose`。
- 若确实要更新，必须在同一处同步 `mAllKeyFrames_[id].world_pose`、`mAllMapPoints_` 里的点、以及 `loop_detector_` 里的登记，否则地图内部就是自相矛盾的。

---

## 四、R3：畸变模型与理想针孔不一致（P1）

### 现状

- `src/sfc_core/src/frontend/tracking.cpp:257`：`robust_pnp_solver_->setCamera(config.K, cv::Mat());` —— **D 传空矩阵**，PnP 内部完全不知道畸变存在。
- `src/sfc_core/src/frontend/tracking.cpp:337-384` `compute3DPoints()`：`disable_undistort=true` 时用**畸变像素** `cvRound(kp.pt)` 取深度，然后 `x = (kp.pt.x - cx) * depth / fx`（理想针孔反投影）。
- `src/sfc_core/src/frontend/tracking.cpp:2350`：`obs = Eigen::Vector2d(curr_kpts[i].pt.x, curr_kpts[i].pt.y)` —— 也是畸变像素。
- 于是「3D 点用畸变像素 + 理想针孔反投影」与「BA 用畸变像素 + 理想针孔投影」**完全自洽**，残差接近 0，真实误差被系统性掩盖。

### 量级

fr1 的 `k1=0.262383`、`k2=-0.953104`、`k3=1.163314` 是强畸变。画面边缘（半径 ≈500 px）径向位移 20–40 px；在 2 m 深度处等价于 8–16 cm 的横向位置误差。这是"每帧都偏、偏得稳定所以看不出来"的那部分误差。

### 修法

**选定：全程走理想像素（`disable_undistort=false` 路线），并把 D 传给 PnP。**

1. `test_tum_rgbd.cpp` 不再强制 `disable_undistort=true`：TUM RGB-D 的深度确实配准在畸变 RGB 坐标系下，所以采样深度时必须用**畸变坐标**，但反投影必须用**理想坐标**。正确顺序是：
   ```
   畸变像素 (u_d,v_d) → 取深度 depth(u_d,v_d)          // 用畸变坐标采样
                     → undistortPoints(..., P=K) 得 (u_i,v_i)   // 转理想坐标
                     → x = (u_i-cx)*d/fx, y = (v_i-cy)*d/fy     // 用理想坐标反投影
   ```
   现有 `:1104-1122` 与 `:2322-2336` 的 `cv::undistortPoints(input, undist, K, D, noArray(), K)` 写法本身是对的，只是被 `disable_undistort=true` 短路掉了。
2. `Tracking` 构造里把 `config_.D` 传给 PnP：`robust_pnp_solver_->setCamera(config.K, config.D);`（第 4 参数现在是空的）。
3. 若短期不想改运行时路径，最低限度必须在 `compute3DPoints` 里对**畸变像素**做一次反畸变再反投影，并在日志里打印「本次用的是畸变还是理想坐标」——不能再让"自洽"掩盖真实偏差。

> **不能**同时做（1）和「边里带畸变模型」两件事，二者叠加会把误差方向翻倍。二选一，写进代码注释。

---

## 四·B、R5：前后端两套观测计数互不相通（P0，与 R1 同级）

> 这一条比 R1 更早就生效，而且**与位姿估计无关**：即使 R1/R2/R3 全部修好，只要 R5 还在，地图仍然是空的。

### 后果推断

`createNewKeyFrame()` 最后一步无条件调 `cleanOldMapPoints()`（`tracking.cpp:3129`）。若 `observed_count` 永远等于 1，则**第一个关键帧创建完的那一刻，整张前端地图的点全部变成 `is_bad = true`**。而 `poseOptimization()` 收集对应的第一道过滤就是 `if (!mp || mp->is_bad) continue;`（`tracking.cpp:2345`）。

推论：第 2 帧之后 `valid.size()` 趋近 0 → `:2355 if (valid.size() < 10) return 0;` 直接返回 → 跟踪退化成纯投影搜索甚至 reloc。**根因不是"跟丢了"，是地图从一开始就没有点。**

### 证据链（四处，全部已核对）

1. **前端计数 `observed_count` 只有一个有效增长点，而它是死的**
   - 创建时固定为 1：`tracking.cpp:311 mp.observed_count = obs_count;`，而 8 个调用点里 7 个传 `1`（`:852`、`:911`、`:1002`、`:1156`、`:1462`、`:2026`、`:3086`），只有 `triangulateNewMapPoints` 的 `:3201` 传 `2`。
   - 唯一的 `++` 在 `checkReplacedInLastFrame()`（`:2785`），依赖 `mReplaceMap_` 非空。
   - `mReplaceMap_` 的唯一写入点是构造函数里的回调（`:288 mReplaceMap_[old_id] = new_id;`）。
   - **`replacement_callback_` 全仓没有任何调用点**（`grep -rn "replacement_callback_("` 零命中）。LocalMapping 线程虽然在跑（`local_mapping.cpp:21 run()`），但从不触发替换。
   - → 纯 VO 下 `mReplaceMap_` 恒空 → `observed_count` **永远停在创建时的 1**。

2. **后端计数 `found_count_` 被写活了，但前端不读它**
   - `addObservationToMapPoint()`（`tracking.cpp:324-332`）操作的是 `mpMap_->getMapPoint(mp_id)`，即 **`mapping::MapPoint`**，调 `mp->addObservation(...)` + `mp->increaseFound()`。
   - `mapping::MapPoint` 的计数 API 是 `found_count_`（`mapping/map_point.hpp:44-47`），**没有 `observed_count`**。
   - `searchLocalPoints()`（`tracking.cpp:2262`）确实调了 `addObservationToMapPoint()`，成功把后端的 `found_count_` 加上去了。
   - 但 `cleanOldMapPoints()` 读的是 **`mAllMapPoints_` 里的 `MapPointCandidate::observed_count`**（`tracking.hpp:108`），与后端 `found_count_` 是两套独立的计数器，**互不同步**。

3. **坏点会永久阻断重建**
   - `createNewKeyFrame()` 建新点前有 `:3062-3065`：`if (i < mvpCurrentMapPointIds_.size() && mvpCurrentMapPointIds_[i] >= 0) { n_points++; continue; }` —— 命中已有 id 就跳过，**不检查 `is_bad`**。
   - `updateLastFrame()` 建点前也有 `:979-982` 同类判断。
   - 于是被 `cleanOldMapPoints()` 判坏的点仍占着 `mvpCurrentMapPointIds_[i]` / `mLastFrameMapPointIds_[i]` 的槽位，新点永远不会被创建，坏点也永不复活（`is_bad` 全仓只有 `:313` 置 false 与 `:3222/:3228` 置 true，没有复活路径）。
   - 净效果：**建图能力在第 1 个关键帧之后归零，且不可恢复。**

### 修法

**选定：让前端计数成为唯一事实来源，把后端计数降级为统计量。**

1. `addObservationToMapPoint()` 里同时增加前端计数：
   ```cpp
   auto* cand = getMapPoint(mp_id);   // mAllMapPoints_ 里的 MapPointCandidate
   if (cand) {
       cand->observed_count++;
       cand->confirmed = (cand->observed_count >= 2);
   }
   ```
   这条补上之后，`searchLocalPoints()` 每成功匹配一次都会让点变成"有冗余观测"，`cleanOldMapPoints()` 才有正确输入。
2. `cleanOldMapPoints()` 增加**只对本次新建点生效**的范围限制，不要遍历全图：现在它每建一个 KF 就 O(N) 扫一遍 `mAllMapPoints_`（13082 点时是每 KF 13k 次哈希遍历）。
3. 建点前的 `mvpCurrentMapPointIds_[i] >= 0` 判断加 `is_bad` 例外：命中坏点时应允许覆盖并重建，否则坏点会永久占位。
4. 建议重命名分清两者：`MapPointCandidate::observed_count` → `frontend_observation_count`；`mapping::MapPoint::found_count_` → `backend_found_count_`。两者语义不同，同名会持续误导（违反 R-026 同理）。

### 验收判据

- 跑 fr1_desk2 前 60 帧，打印 `mAllMapPoints_.size()` 与 `is_bad` 数量：修前 `is_bad == size()`（第一个 KF 后立刻全坏），修后 `is_bad / size() < 0.3`。
- 同一段日志里 `poseOptimization` 的 `valid.size()` 应稳定在数百量级，而不是掉到 10 以下。

---

## 五、R4：次级问题（R1–R5 修完再动，否则无法归因）

| 项 | 位置 | 现状 | 建议 |
|---|---|---|---|
| 观测权重不分层级 | `tracking.cpp:2462-2467` | `int level` 算了不用，`invSigma2` 恒 `1/1.44` | `invSigma2 = 1.0f / (sigma2 * pow(scale_factor, 2*level))` |
| 地图点只建不删 | `tracking.cpp:3218-3231` | `observed_count<=1` 只置 `is_bad`，从不 `erase`；13082 个点 | 临时点不进 `mAllMapPoints_`；`retireBadMapPoints()` 真正删除并清理悬空 id |
| 单帧阈值过宽 | `tracking.cpp:1210-1211`（PnP 5.0 m / 1.5 rad）、`:1408-1416`（几何容差 20 px）、`:2818-2819`（速度 1.0 m/帧） | 真实 30 Hz 手持 3–7 cm/帧 | 分别收到 0.3 m / 0.3 rad、8 px、0.15 m/帧 |
| `trackWithMotionModel` 失败不恢复初值 | `tracking.cpp:1016-1061` | 失败时 `mCurrentFrame_->pose` 仍是错误预测值，SBP 回退分支直接用它 | 失败时恢复为 `mLastFrame_->pose` |
| 复用分支不检查 `is_bad` | `tracking.cpp:1451-1466` | 复用 `mLastFrameMapPointIds_` 时不判断坏点 | 加 `if (!mp \|\| mp->is_bad) continue;` |
| `mvpLocalKeyFrames_` 混入普通帧 | `tracking.cpp:2090-2128` | `updateLocalMap(true)` 把非关键帧也 push 进去，`:572`/`:1077`/`:2845` 三处 `back()` 当参考帧 | 只放关键帧；普通帧需要参考时用 `mLastFrame_` |
| `Tlr` 被赋成绝对位姿 | `updateLastFrame()` 内 `const Pose& Tlr = mLastFrame_->pose;` | 变量名是"相对变换"，实际是绝对位姿，读代码必被误导 | 改名 `Tlw`，或真的算 `Tlr` |
| `bow_matches` 索引越界 | `tracking.cpp:1149` `pts2d.emplace_back(curr_kps[i_f]...)` | 用 `bow_matches.size()` 索引 `curr_kps`，无越界检查 | 加 `if (i_f >= (int)curr_kps.size()) continue;` |

---

## 六、修复顺序（E1 → E7，每步单独可测）

| 步 | 内容 | 验收判据 |
|---|---|---|
| **E1** | **R5**：`addObservationToMapPoint()` 同步前端 `observed_count`；建点时允许覆盖坏点 | 第 60 帧时 `is_bad / mAllMapPoints_.size() < 0.3`；`valid.size()` 稳定在数百 |
| **E2** | R1 方案 A：`RobustPnPSolver` 出口统一转 T_wc；`poseOptimization` 显式 `getCwFromPose` 建顶点、取回后转回 | f=1..10 的 `pose` 不再逐帧翻号；`delta_t` 回落到 0.03–0.08 m 量级 |
| **E3** | R2：删除 `updateLastFrame()` 里的位姿覆盖 | `[DBG n] pose` 与下一帧 `last_pose` 完全一致（差值 < 1e-6） |
| **E4** | R3：全程理想像素 + 把 D 传给 PnP | `init_reproj_err_avg` 在 f=0..10 稳定 < 1.5 px；边缘特征的内点数上升 |
| **E5** | R4 阈值收紧 | 单帧 `\|t\|` 中位数落到 0.03–0.07 m；`REJECTED` 出现频率 < 2% |
| **E6** | R4 地图点生命周期（临时点池 + 真正 erase + 近邻查重） | `mAllMapPoints_.size()` 不再随帧数单调增长（fr1_desk2 < 3000） |
| **E7** | 回填 `docs/PLAN-slam-loop-pgo-fix.md` 的 D1–D7（回环层） | 回环层此时才有意义 |

> **顺序不可调**：E1–E3 未修之前接回环，等于在错误地图上加约束——这正是当前 1165 m 的来源。

---

## 七、验收标准

用 `scripts/evaluate_trajectory.py`（R-024），**RGB-D 不估尺度**：

```bash
# 1) 跑纯 VO
./build/sfc_core/test_tum_rgbd --dataset ~/datasets/tum/rgbd_dataset_freiburg1_desk \
    --output /tmp/fr1_desk2_vo --max_frames 640

# 2) 评测（--align 浅对齐，不带 --align-scale）
python3 scripts/evaluate_trajectory.py \
    --est /tmp/fr1_desk2_vo.tum \
    --ref ~/datasets/tum/rgbd_dataset_freiburg1_desk/groundtruth.txt \
    --align
```

| 指标 | 当前（纯 VO） | E1–E5 后目标 |
|---|---|---|
| ATE rmse | 未测（路径总长 478.6 m，真值 5–12 m） | **≤ 0.10 m** |
| RPE_trans rmse（Δ=1） | 未测 | **≤ 0.03 m** |
| 路径总长 | 478.6 m | 8–14 m |
| 单帧位移中位数 | ≈0.75 m | 0.03–0.07 m |
| 起终点漂移 | 4.300 m | ≤ 0.15 m |
| 640 帧总耗时 | ≈50 ms/帧 | 保持 < 100 ms/帧 |

---

## 八、风险与回滚

- 项目**不是 git 仓库**，改动前逐个文件 `cp` 到 `/tmp/`：
  ```bash
  cp src/sfc_core/src/frontend/tracking.cpp       /tmp/tracking.base.cpp
  cp src/sfc_core/src/frontend/robust_pnp_solver.cpp /tmp/pnp.base.cpp
  cp src/sfc_core/test/test_tum_rgbd.cpp          /tmp/test_tum_rgbd.base.cpp
  ```
- E1 会一次性改变全系统位姿语义，**必然引起其他模块（PGO / 回环 / mapping）行为变化**。建议 E1 落地后先只跑纯 VO（不加 `--slam`），确认前端正确再开回环。
- `KeyFrameCacheEntry::world_pose`、`mpMap_->getKeyFrame()->setPose()`、`loop_detector_->updateKeyFramePose()` 三处会随 E1 一起翻转，需要在同一改动里核对。
- 若 E1 后误差反而变大，说明真实原因是 R3（畸变），此时先单独 revert E1 再验 R3 —— 不要两个一起改然后猜。

---

## 九、与 CODE-RULES 的关系

本次诊断暴露两条现有规范未被遵守，落地时一并处理：

- **R-020（位姿用类型封装、显式区分世界系/相机系）** —— 本次 R1 正是该规则的直接反例：`Pose` 只有一个 struct，`q/t` 到底是 T_wc 还是 T_cw 全靠注释。修 E1 时建议顺带引入 `struct PoseWc` / `struct PoseCw` 两个强类型（或 `Sophus::SE3d` + 命名 `T_wc`/`T_cw`），让混用在编译期就失败。
- **R-004（命名 camelCase）** —— 现有代码 `mVelocity_`、`mnMatchesInliers_`、`mvpLocalKeyFrames_` 混用匈牙利前缀；可读性整改阶段统一。

> 建议新增 **R-026**：任何跨模块传递的位姿必须在类型或变量名上标明参考方向（`T_wc` / `T_cw`），不得用裸 `Pose` 传递而不注明；OpenCV PnP 与 g2o `VertexSE3Expmap` 的返回值一律视为 `T_cw`，与系统内部 `T_wc` 之间只允许在单一出入口转换一次。

---

## 十、遗留

- 本机（Windows）无 C++ 工具链，全部改动仍需用户在 Ubuntu `/home/syf/SFC-SLAM` 上 `colcon build` 验证。
- 阶段 C 的 ATE/RPE 从未实跑过（无数据集），E1 之后是第一次真正的量化验收。
- `docs/PLAN-tracking-drift-fix.md`（纯 VO 阶段 A/B1/B3 记录）与 `docs/PLAN-slam-loop-pgo-fix.md`（回环层）保留不动，本文件是第三份、也是优先级最高的一份。


---

## 十一、执行记录（2026-11-10）

**范围**：E1(R5) → E2(R1) → E3(R2) → E4(R3) → E5/E6(R4)，全部按本文件规定的顺序落地。
**验证状态**：本机为 Windows，无 cmake/g++/clang/MSVC/ninja/WSL，**一行都没有编译过**。
只做了静态核对：括号配对（`scripts/check_braces.js`）+ 遗留符号扫描（`scripts/static_check_e6.js`），
五个前端文件 `final brace=0 paren=0 brack=0`，`observed_count` / `R_eigen` / `need_undistort` 均 0 处。

### E1 / R5 · 前后端观测计数打通

- `MapPointCandidate::observed_count` 更名 `frontend_observation_count`，新增 `created_frame_id` /
  `last_observed_frame_id`；`addObservationToMapPoint()` 现在同时写前端候选（原来只写后端
  `mapping::MapPoint`，前端计数永远停在创建值）。
- `cleanOldMapPoints()` 只在 `mNewMapPointIds_`（观察期内的新建点）上判定，加了
  `kMapPointObservationGraceFrames = 3` 的宽限期；原来它遍历全表、把 `observed_count<=1` 的点
  一律标坏，第 1 个关键帧之后整张图就废了。
- 建点前的"已有 id 就跳过"判断补上 `is_bad` 例外，坏点不再永久占位。

### E2 / R1 · 位姿约定统一（T_wc vs T_cw）

- 选定**方案 A**：`RobustPnPSolver` 内部一律 `T_cw`（OpenCV PnP / g2o 顶点的语义），
  只在出入口各转换一次 —— 新增静态方法 `poseCwToWc()` / `poseWcToCw()`，
  `optimize()` 入口把 `init_pose` 转成 `T_cw`、出口把结果转回 `T_wc`，`mlpnpSolve()` 同理。
- `poseOptimization()`：建 `VertexSE3Expmap` 前显式 `getCwFromPose(p)`，取回后用新增的
  `poseFromCw()` 转回 `T_wc`，再写 `mCurrentFrame_->pose`。
- `covis_manager.cpp` 的局部 BA 同属一个位姿约定，一并改：新增
  `poseWcToSe3Quat()` / `se3QuatToPoseWc()`，顶点建/取回都过这两个函数。
- 全文件已无 `R_eigen` 残留（那是把 `T_wc` 直接塞进 `T_cw` 顶点的那行）。

### E3 / R2 · 删除 `updateLastFrame()` 的速度预测覆盖

- 整段 `mLastFrame_->pose = updated_pose;` + `updateWorldPoints(mLastFrame_)` 删除，
  换成说明注释：上一帧位姿是 `poseOptimization` 的优化结果，用恒速外推覆盖等于每帧丢掉
  优化值、误差按帧累积。
- 顺带把误导性的局部变量名 `Tlr` 改成 `last_frame_pose`（它只是上一帧的绝对位姿，
  不是帧间相对变换）。

### E4 / R3 · 畸变（**偏离计划字面执行**）

计划 §四 同时要求"全程理想像素"和"`setCamera(config.K, config.D)`"，但同节末行明确
告警"不能同时做这两件事"。实读代码后确认：`trackFrame()` 已经在 `disable_undistort=false`
时把 2D 观测去畸变成理想像素，若再把 `config.D` 传给 PnP，OpenCV 会对理想像素二次去畸变。

**最终选择"全程理想像素"这一支，PnP 保持接收空 `D`**，理由写进了构造函数注释。

真正的修复在 `compute3DPoints()`：原来"采样深度"与"反投影"共用同一套坐标，
于是"畸变像素 + 理想针孔"在 3D 与 2D 两侧自洽，误差被掩盖。现在两件事拆开 ——
**采样用畸变像素**（深度图与畸变 RGB 对齐）、**反投影用理想像素**（相机模型是理想针孔），
索引始终对齐 `frame->keypoints`。

配套：`config/tracking.yaml` 的 `disable_undistort` 改 `false`；`test_tum_rgbd.cpp` 不再强制
覆盖该字段；`test_visual_slam.cpp` 两处改为 `tcfg.disable_undistort = tcfg.D.empty()`；
各调用点删掉"对已是理想像素的 `undistorted_keypoints` 再做一次 `undistortPoints`"的重复处理
（`trackFrame()` 的 Step 4 成为唯一的坐标系决定点）。

### E5 / R4 · 阈值与流程

- 观测权重按特征点所在金字塔层加权（`invSigma2 = 1/(1.2·scale^level)^2`），不再固定 1.2。
- PnP 一致性阈值 5.0m / 1.5rad → **0.3m / 0.3rad**（原值等于"PnP 永远通过"）。
- `trackWithMotionModel()` 失败时把位姿退回本帧起点（原来留着恒速预测的坏初值给后续路径）。
- `updateLocalMap()` 不再把普通帧塞进 `mvpLocalKeyFrames_`（关键帧窗口只由
  `createNewKeyFrame()` 维护），避免 `trackReferenceKeyFrame()` 拿本帧自己的
  `poseOptimization` 结果当参考、误差自我循环。
- `bow_matches` 循环补 6 条越界/无效闸；几何一致性容差 `max(radius*2.5, 20px)` → `max(radius*1.5, 8px)`；
  运动模型合理性阈值 1.0 m/帧 → 0.15 m/帧。

### E6 · 地图点生命周期

- 临时 VO 点改为进独立池 `mTemporalMapPoints_`，与主表共用 `next_mp_id_` 空间；
  `getMapPoint()` 先查主表再查临时池。原来它们照样进 `mAllMapPoints_` 且永不回收。
- `retireBadMapPoints()` 真正 `erase` 坏点，并清理 `mvpCurrentMapPointIds_` /
  `mLastFrameMapPointIds_` / 所有关键帧 `map_point_ids` 里的悬空 id。
  调用点固定在单帧流程末尾、无 `MapPointCandidate*` 被持有的时点。
- `createMapPoint()` 先做体素近邻查重（`kMapPointVoxelSize = 0.10` / `kMapPointDuplicateRadius = 0.05`），
  命中则复用已有 id 而不是新建。

### 未完成 / 需在有工具链的机器上确认

1. **编译**：全部改动未经编译。C++17 已确认（`src/sfc_core/CMakeLists.txt:20` `set(CMAKE_CXX_STANDARD 17)`），`std::clamp` 可用；`<algorithm>` / `<cmath>` / `<cstdint>` 三个新依赖的头文件均已在位。
2. **ATE/RPE**：C1 的评测脚本存在，但从未实跑（无数据集）。
3. `updateLocalMap()` 删掉普通帧后，`mpReferenceKF_` 的语义变化需要在真实序列上看
   `trackReferenceKeyFrame()` 的命中率是否下降。
4. R-020 建议的两个强类型 `PoseWc` / `PoseCw` 未引入 —— 本次只统一了转换点，没有让混用
   在编译期失败。这是本文件 §九 给出的建议，不是 E1 的验收条件。
