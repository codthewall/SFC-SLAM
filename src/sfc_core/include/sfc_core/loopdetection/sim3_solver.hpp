// 必须在最开头清除 CMake project() 注入的 sfc_core 宏（g2o CMake 会注入 -Dsfc_core=1）
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_LOOPDETECTION_SIM3_SOLVER_HPP_
#define SFC_CORE_LOOPDETECTION_SIM3_SOLVER_HPP_

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace sfc_core {
namespace loopdetection {

// ================================================================
// Sim3Solver: 3-point RANSAC + Horn 1987 闭式 Sim3 求解器
//
// 参考 ORB_SLAM3/src/Sim3Solver.cc，完全对齐其行为。
//
// 用途：RGB-D 回环检测时，从两帧的共视地图点对应关系中
//       鲁棒地估计 Sim3（旋转 + 平移 + 尺度）相对变换。
//
// 与手写 ICP RANSAC 的区别：
//   - 手写 ICP：3-point SVD + 重投影误差 RANSAC（不稳定，易失败）
//   - Sim3Solver：Horn 1987 闭式解（RANSAC 采样 -> 闭式求解 -> 重投影验证）
//     闭式解比迭代 SVD 更稳定，不依赖初始值。
//
// 关键设计：
//   - mbFixScale = true（RGB-D 有真实尺度，不估计尺度）
//   - CheckInliers 使用相机内参做重投影误差验证
//   - 最小采样数 = 3（与 ORB_SLAM3 一致）
// ================================================================

class Sim3Solver {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // kf1_pts3d_cam: 关键帧1 在其相机坐标系下的 3D 点（相机系坐标）
    // kf2_pts3d_cam: 关键帧2 在其相机坐标系下的 3D 点（相机系坐标）
    // matches: queryIdx -> kf1 点索引，trainIdx -> kf2 点索引
    // K: 相机内参（用于 CheckInliers 重投影误差验证）
    // fixScale: true = RGB-D 模式（固定尺度 1.0），false = 单目模式（估计尺度）
    Sim3Solver(const std::vector<Eigen::Vector3d>& kf1_pts3d_cam,
               const std::vector<Eigen::Vector3d>& kf2_pts3d_cam,
               const std::vector<cv::DMatch>& matches,
               const cv::Mat& K,
               bool fixScale = true);

    // RANSAC 参数设置
    // probability: 期望的成功概率（默认 0.99）
    // minInliers: 最少内点数（RANSAC 停止条件）
    // maxIterations: 最大迭代次数
    void setRansacParameters(double probability = 0.99, int minInliers = 6, int maxIterations = 300);

    // RANSAC 主迭代
    // nIterations: 每次调用最多迭代次数
    // bNoMore[out]: 迭代耗尽时置为 true
    // vbInliers[out]: 内点标记（与输入 matches 索引对应）
    // nInliers[out]: 内点总数
    // 返回: 4x4 Sim3 矩阵（T12: 帧2 -> 帧1 的相对变换，尺度 = 1.0）
    Eigen::Matrix4f iterate(int nIterations, bool& bNoMore,
                             std::vector<bool>& vbInliers, int& nInliers);

    // 简单接口（使用全部 maxIterations）
    Eigen::Matrix4f find(std::vector<bool>& vbInliers, int& nInliers);

    Eigen::Matrix3d getEstimatedRotation() const { return mR12i.cast<double>(); }
    Eigen::Vector3d getEstimatedTranslation() const { return mt12i.cast<double>(); }
    float getEstimatedScale() const { return ms12i; }
    Eigen::Matrix4d getEstimatedTransformation() const { return mT12i.cast<double>(); }

private:
    // 质心计算
    void computeCentroid(Eigen::Matrix3f& P, Eigen::Matrix3f& Pr, Eigen::Vector3f& C);

    // Horn 1987 闭式 Sim3 求解
    // P1, P2: 去质心后的 3x3 矩阵（每列是一个 3D 点，共 3 列）
    // 输出: mR12i（旋转）, mt12i（平移）, ms12i（尺度）, mT12i（4x4 矩阵）
    void computeSim3(Eigen::Matrix3f& P1, Eigen::Matrix3f& P2);

    // 使用当前估计的 Sim3 验证所有匹配，计算内点
    // 内点判定: 重投影误差 < mvnMaxError（基于特征点尺度层数）
    void checkInliers();

    // 将 3D 点用 Sim3 投影到图像平面
    void project(const std::vector<Eigen::Vector3f>& vP3Dw,
                 std::vector<Eigen::Vector2f>& vP2D,
                 const Eigen::Matrix4f& Tcw) const;

    // 将 3D 点从相机坐标系投影到图像平面
    void fromCameraToImage(const std::vector<Eigen::Vector3f>& vP3Dc,
                           std::vector<Eigen::Vector2f>& vP2D) const;

    // 相机参数（从 K 矩阵提取）
    float fx_, fy_, cx_, cy_;

    // 输入数据
    std::vector<Eigen::Vector3d> mvX3Dc1_;
    std::vector<Eigen::Vector3d> mvX3Dc2_;
    std::vector<cv::DMatch> mMatches_;
    std::vector<size_t> mvAllIndices_;
    std::vector<size_t> mvnIndices1_;
    std::vector<float> mvnMaxError1_;
    std::vector<float> mvnMaxError2_;

    // 预计算的图像坐标（用于 CheckInliers）
    std::vector<Eigen::Vector2f> mvP1im1_;
    std::vector<Eigen::Vector2f> mvP2im2_;

    // RANSAC 状态
    int mnIterations_ = 0;
    int mRansacMaxIts_ = 300;
    float mRansacProb_ = 0.99f;
    int mRansacMinInliers_ = 6;
    int N_ = 0;

    // 当前估计
    Eigen::Matrix3f mR12i;
    Eigen::Vector3f mt12i;
    float ms12i = 1.0f;
    Eigen::Matrix4f mT12i;
    Eigen::Matrix4f mT21i;
    std::vector<bool> mvbInliersi_;
    int mnInliersi_ = 0;

    // 最佳结果
    Eigen::Matrix3f mBestRotation_;
    Eigen::Vector3f mBestTranslation_;
    float mBestScale_ = 1.0f;
    Eigen::Matrix4f mBestT12_;
    std::vector<bool> mvbBestInliers_;
    int mnBestInliers_ = 0;

    bool mbFixScale_ = true;
};

}  // namespace loopdetection
}  // namespace sfc_core

#endif  // SFC_CORE_LOOPDETECTION_SIM3_SOLVER_HPP_
