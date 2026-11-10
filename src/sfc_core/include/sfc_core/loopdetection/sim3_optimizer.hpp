// 必须在最开头清除 CMake project() 注入的 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_LOOPDETECTION_SIM3_OPTIMIZER_HPP_
#define SFC_CORE_LOOPDETECTION_SIM3_OPTIMIZER_HPP_

#include <Eigen/Dense>
#include <vector>
#include <opencv2/opencv.hpp>

namespace sfc_core {
namespace loopdetection {

// ================================================================
// Sim3Optimizer: g2o 7DoF Sim3 优化器
//
// 参考 ORB_SLAM3/src/Optimizer.cc 的 OptimizeSim3() 实现。
//
// 用途：在 Sim3Solver RANSAC 找到初始 Sim3 后，
//       用 g2o 联合优化 Sim3 顶点和地图点顶点，
//       进一步精化相对位姿估计。
//
// 两轮精化策略（与 ORB_SLAM3 完全一致）：
//   第一轮：粗匹配，阈值 8px，RANSAC 300 次，OptimizeSim3 10 次迭代
//   第二轮：精匹配，阈值 5px，OptimizeSim3 10 次迭代
// ================================================================

// 地图点匹配对（用于 Sim3Solver 和 Sim3Optimizer）
struct MapPointMatch {
    int kf1_kpt_idx;   // 关键帧1 的特征点索引
    int kf2_kpt_idx;   // 关键帧2 的特征点索引（可为 -1）
    Eigen::Vector3d pt_world;  // 地图点的世界坐标
    Eigen::Vector3d pt_cam1; // 关键帧1 相机系坐标（已废弃，仅为兼容）
    Eigen::Vector3d pt_cam2; // 关键帧2 相机系坐标（已废弃，仅为兼容）
    // ★★★ 原始像素坐标：用于 g2o Sim3 优化的观测值 ★★★
    // 必须使用原始检测到的像素坐标，而非重投影的相机系坐标
    // 这避免了「用相机系坐标重投影作为观测」导致的循环依赖
    Eigen::Vector2d px1;  // 关键帧1 的原始检测像素坐标
    Eigen::Vector2d px2;  // 关键帧2 的原始检测像素坐标
};

class Sim3Optimizer {
public:
    // 两帧之间的 Sim3 精化（参考 ORB_SLAM3 Optimizer::OptimizeSim3）
    // kf1: 关键帧1（当前帧）
    // kf2: 关键帧2（匹配帧 / 回环候选帧）
    // matches: 两帧之间的地图点匹配对
    // initial_sim3: 初始 Sim3 估计（T12: 帧2 -> 帧1 的变换）
    // K: 相机内参
    // fix_scale: true = RGB-D（固定尺度），false = 单目（估计尺度）
    // T_cw1 / T_cw2: 世界到相机变换矩阵，用于深度检查（pt_world -> camera coords）
    //   T_cw = [R_cw | t_cw; 0 0 0 1]，其中 R_cw * P_world + t_cw = P_camera
    //   注意：必须传入真实的相机位姿，而非单位矩阵
    // 返回：精化后的内点数（< 15 则认为失败）
    static int optimizeSim3(const Eigen::Matrix4d& initial_sim3,
                          const std::vector<MapPointMatch>& matches,
                          const cv::Mat& K,
                          bool fix_scale,
                          const Eigen::Matrix4d& T_cw1,
                          const Eigen::Matrix4d& T_cw2,
                          Eigen::Matrix4d& out_sim3);

    // 两轮精化主函数（与 ORB_SLAM3 LoopClosing.cc DetectCommonRegionsFromBoW 对齐）
    // first_th: 第一轮阈值（像素）
    // second_th: 第二轮阈值（像素）
    // max_iterations: 每轮 OptimizeSim3 最大迭代次数
    // T_cw1 / T_cw2: 世界到相机变换矩阵
    // 返回：精化后的 Sim3
    static int refineWithTwoPass(const Eigen::Matrix4d& initial_sim3,
                                 const std::vector<MapPointMatch>& matches,
                                 const cv::Mat& K,
                                 bool fix_scale,
                                 float first_th,
                                 float second_th,
                                 int max_iterations,
                                 const Eigen::Matrix4d& T_cw1,
                                 const Eigen::Matrix4d& T_cw2,
                                 Eigen::Matrix4d& out_sim3);
};

}  // namespace loopdetection
}  // namespace sfc_core

#endif  // SFC_CORE_LOOPDETECTION_SIM3_OPTIMIZER_HPP_
