# SFC-SLAM

视觉 SLAM 系统：ORB-SLAM3 风格的 C++17 核心库 `sfc_core` + ROS 2 Humble 接口包 `sfc_ros_humble`，默认以 TUM RGB-D 序列做精度验收。

- 核心算法库：`src/sfc_core/`（特征提取、光流/匹配、PnP 位姿求解、局部地图、回环检测与位姿图优化）
- ROS 2 接口：`src/sfc_ros_humble/`（话题订阅、TF 发布、可视化与 launch）
- 配置：`config/*.yaml`（唯一参数来源）
- 开发期工具：`scripts/`（静态检查、批量改写、轨迹评测、数据回放）
- 诊断与修复计划：`docs/PLAN-*.md`
- 代码约束规则：`CODE-RULES.md`（改代码前必读）

## 依赖

| 类别 | 组件 |
| --- | --- |
| 编译 | CMake ≥ 3.16、支持 C++17 的编译器 |
| 数学 | Eigen3、Sophus |
| 视觉 | OpenCV（core / imgproc / features2d / calib3d） |
| 优化 | g2o、SuiteSparse（Linux 走系统包，Windows 走预编译） |
| 视觉词袋 | DBoW2（`src/sfc_core/*/thirdparty/dbow2`，已内联） |
| 可选 | ROS 2 Humble（仅 `sfc_ros_humble` 需要） |

### 词袋文件（不入库）

`ORBvoc.txt` 约 145 MB，超出 Git 单文件限制，**不提交**，按落地位置自行放置：

```bash
# 放到 config/ 与 Thirdparty/ 下（两处都需要，供不同入口加载）
cp ORBvoc.txt config/ORBvoc.txt
cp ORBvoc.txt Thirdparty/ORBvoc.txt
```

来源：ORB-SLAM3 官方词袋 `ORBvoc.txt.tar.gz` 解压后即为该文件。

## 构建

```bash
# 核心库
cmake -S src/sfc_core -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j

# ROS 2 工作区（可选）
colcon build --packages-select sfc_core sfc_ros_humble
```

## 运行与评测

```bash
# TUM RGB-D 回放（脚本封装了 rosbag 播放与帧同步）
bash scripts/run_and_eval_tum.sh

# 轨迹精度：ATE / RPE（唯一验收口径，见 R-024）
python scripts/evaluate_trajectory.py --est CameraTrajectory.txt --gt groundtruth.txt
```

RGB-D 与双目序列**不要**开启对齐估尺度——尺度来自深度/基线，估尺度会把尺度误差藏进对齐变换里。

## 工程流程

本仓库以 GitHub 为唯一主干，代码与规则文档都在 GitHub 上维护：

1. **改代码前**读 `CODE-RULES.md`，选中本次要守的规则（文件规模、命名、配置单一来源、诊断日志开关、ATE/RPE 验收）。
2. **改动 → 提交 → 推送**：每一处改动都要在 GitHub 留下痕迹，不允许只存在于本地工作区。
3. **提交信息**：`<type>(<scope>): <subject>`，其中 `type ∈ feat|fix|refactor|perf|test|docs|chore|rules`；`fix`/`perf` 必须带 `验证:` 行写明测过的命令与结果（ATE/RPE 数值、测试用例）。
4. **规则变更**：调整 `CODE-RULES.md` 时同时追加一行「变更记录」（日期、版本号、改了什么、为什么改），并用 `rules_update` 写入，不手改正文。
5. **分支**：`main` 保持可编译可评测；较大改动走 `feat/<topic>`、`fix/<topic>` 分支后合并。

### 提交示例

```
fix(frontend): 统一 T_cw / T_wc 约定，修复位姿整体取反

根因：RobustPnPSolver::optimize 输出 T_cw 被直接当作 T_wc 存进
mCurrentFrame_->pose，导致纯 VO 每帧平移反向。

验证: test_tum_rgbd fr1_desk2, 无 --align 估尺度, ATE rmse 0.31 m
      (修复前 5.42 m)；ctest 5/5 通过
```

## 目录结构

```
CODE-RULES.md          代码约束规则（含变更记录）
config/                yaml 配置与相机内参
docs/                  PLAN-*.md 诊断与修复计划
picture/               文档配图
scripts/               开发期脚本（JS/Python/Shell）
src/sfc_core/          核心算法库（include/ + src/ + test/ + Thirdparty/）
src/sfc_ros_humble/    ROS 2 Humble 接口包
.rules/                CODE-RULES 的历史版本
```
