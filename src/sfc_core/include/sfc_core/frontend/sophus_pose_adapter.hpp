/**
 * SophusPoseAdapter - Sophus SE3 位姿适配器
 * 使用 Sophus::SE3f 作为内部表示，同时兼容 SFC-SLAM 的 Pose 接口
 */

// 必须在最开头执行：清除 CMake project() 注入的 sfc_core 宏
// 原因：g2o 的 CMakeLists.txt 设置了 g2o_DIR，且 project(sfc_core) 会生成 -Dsfc_core=1 宏。
// 若不提前 undef，标准库头文件（如 <unordered_set>）会被解析为:
//   std::unordered_set → sfc_core_1::frontend::std::unordered_set
// 从而破坏 std::hash、std::unordered_set 等模板的正常查找。
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_FRONTEND_SOPHUS_POSE_ADAPTER_HPP_
#define SFC_CORE_FRONTEND_SOPHUS_POSE_ADAPTER_HPP_

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "sophus/geometry.hpp"

namespace sfc_core {
namespace frontend {

/**
 * SophusPose - ORB-SLAM3 风格的位姿表示
 * 
 * 使用 Sophus::SE3f 作为内部存储，提供与 SFC-SLAM Pose 兼容的接口。
 * 主要优势：
 * 1. 数学运算更简洁（使用 Lie 代数）
 * 2. 自动处理旋转矩阵的正交性
 * 3. 插值和扰动计算更方便
 */
class SophusPose {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    SophusPose();
    SophusPose(const Sophus::SE3f& se3);
    SophusPose(const Eigen::Matrix3f& R, const Eigen::Vector3f& t);
    SophusPose(const Eigen::Quaternionf& q, const Eigen::Vector3f& t);
    
    // 从传统 Pose 构造
    static SophusPose fromPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);
    
    // 转换为传统 Pose
    Eigen::Quaterniond toQuaterniond() const;
    Eigen::Vector3d toTranslationd() const;
    
    // 获取内部 SE3
    const Sophus::SE3f& se3() const { return se3_; }
    Sophus::SE3f& se3() { return se3_; }
    
    // 基本运算
    SophusPose inverse() const;
    Eigen::Vector3f transformPoint(const Eigen::Vector3f& point) const;
    Eigen::Vector3f transformPointInverse(const Eigen::Vector3f& point) const;
    
    // 复合运算
    SophusPose operator*(const SophusPose& other) const;
    
    // 李代数扰动
    void applyPerturbation(const Eigen::Vector3f& delta_translation,
                          const Eigen::Vector3f& delta_rotation);
    
    // 获取旋转和平移
    Eigen::Matrix3f rotationMatrix() const { return se3_.rotationMatrix(); }
    Eigen::Quaternionf quaternion() const { return se3_.unit_quaternion(); }
    Eigen::Vector3f translation() const { return se3_.translation(); }
    
    // 设置
    void setRotationMatrix(const Eigen::Matrix3f& R);
    void setQuaternion(const Eigen::Quaternionf& q);
    void setTranslation(const Eigen::Vector3f& t);
    
    // 转换为 cv::Mat (4x4)
    cv::Mat toCvMat4x4f() const;
    cv::Mat toCvMat4x4d() const;
    
    // 判断有效
    bool isValid() const;
    
private:
    Sophus::SE3f se3_;
};

/**
 * PoseConverter - Pose 和 Sophus::SE3f 之间的转换工具
 */
class PoseConverter {
public:
    // 将 Pose (quaternion + vector) 转换为 Sophus::SE3f
    static Sophus::SE3f toSE3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);
    
    // 将 Sophus::SE3f 转换为 Pose (quaternion + vector)
    static void toPose(const Sophus::SE3f& se3, Eigen::Quaterniond& q, Eigen::Vector3d& t);
    
    // 将 cv::Mat (4x4) 转换为 Sophus::SE3f
    static Sophus::SE3f toSE3(const cv::Mat& T);
    
    // 将 Sophus::SE3f 转换为 cv::Mat (4x4)
    static cv::Mat toCvMat(const Sophus::SE3f& se3);
    
    // 计算相对位姿：T_rel = T1^-1 * T2
    static Sophus::SE3f computeRelativePose(const Sophus::SE3f& T1, const Sophus::SE3f& T2);
};

} // namespace frontend
} // namespace sfc_core

#endif // SFC_CORE_FRONTEND_SOPHUS_POSE_ADAPTER_HPP_
