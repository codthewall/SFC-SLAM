// 必须在最开头清除 g2o CMake 注入的 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_FRONTEND_POSE_EKF_HPP_
#define SFC_CORE_FRONTEND_POSE_EKF_HPP_

#include <Eigen/Dense>
#include "frame.hpp"

namespace sfc_core {
namespace frontend {

class PoseEKF {
public:
    PoseEKF();

    void init(const Pose& initial_pose);
    void predict(double dt);
    void update(const Pose& observation);
    Pose getCurrentPose() const;

private:
    Eigen::Matrix<double, 13, 1> x_;       // 状态向量 [p, v, q]
    Eigen::Matrix<double, 13, 13> P_;      // 状态协方差
    Eigen::Matrix<double, 13, 13> Q_;      // 过程噪声
    Eigen::Matrix<double, 7, 7> R_;        // 观测噪声
    bool is_initialized_ = false;
    double last_timestamp_ = 0.0;
};

} // namespace frontend
} // namespace sfc_core

#endif // SFC_CORE_FRONTEND_POSE_EKF_HPP
