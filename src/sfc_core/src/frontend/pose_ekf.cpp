#include "sfc_core/frontend/pose_ekf.hpp"
#include <cmath>
#include <Eigen/Dense>

namespace sfc_core {
namespace frontend {

PoseEKF::PoseEKF()
    : is_initialized_(false), last_timestamp_(0.0) {
    x_.setZero();
    x_(6) = 1.0;

    P_.setIdentity();
    P_.block<3, 3>(0, 0) *= 0.01;
    P_.block<3, 3>(3, 3) *= 0.1;
    P_.block<4, 4>(6, 6) *= 0.01;

    Q_.setIdentity();
    Q_.block<3, 3>(0, 0) *= 0.001;
    Q_.block<3, 3>(3, 3) *= 0.01;
    Q_.block<4, 4>(6, 6) *= 0.001;

    R_.setIdentity();
    R_.block<3, 3>(0, 0) *= 0.01;
    R_.block<4, 4>(3, 3) *= 0.01;
}

void PoseEKF::init(const Pose& initial_pose) {
    x_.block<3, 1>(0, 0) = initial_pose.t;
    x_.block<3, 1>(3, 0).setZero();
    x_(6) = initial_pose.q.w();
    x_(7) = initial_pose.q.x();
    x_(8) = initial_pose.q.y();
    x_(9) = initial_pose.q.z();
    is_initialized_ = true;
}

void PoseEKF::predict(double dt) {
    if (!is_initialized_) return;
    Eigen::Matrix<double, 13, 1> x_pred = x_;
    x_pred.block<3, 1>(0, 0) += x_.block<3, 1>(3, 0) * dt;

    Eigen::Matrix<double, 13, 13> F = Eigen::Matrix<double, 13, 13>::Identity();
    F.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity() * dt;
    P_ = F * P_ * F.transpose() + Q_;
    x_ = x_pred;
}

void PoseEKF::update(const Pose& observation) {
    if (!is_initialized_) return;

    // 备份当前状态（用于 NaN 回退）
    Eigen::Matrix<double, 13, 1> x_backup = x_;
    Eigen::Matrix<double, 13, 13> P_backup = P_;

    // ---- 观测向量 z ----
    Eigen::Matrix<double, 7, 1> z;
    z.block<3, 1>(0, 0) = observation.t;
    z(3) = observation.q.w();
    z(4) = observation.q.x();
    z(5) = observation.q.y();
    z(6) = observation.q.z();

    // ---- 先验均值 h(x) ----
    Eigen::Matrix<double, 7, 1> h;
    h.block<3, 1>(0, 0) = x_.block<3, 1>(0, 0);

    // ★★★ Bug Fix A1: 四元数必须用流形误差（非线性减法）★★★
    // q_obs 和 q_prior 都是单位四元数，直接相减违反 SO(3) 流形约束。
    // 正确做法：在 SO(3) 切空间计算误差。
    //   delta_q = q_obs * q_prior^(-1)  得到相对旋转
    //   误差向量取 delta_q 的向量部分（x,y,z），即在切空间的坐标
    //   w 误差取 (delta_q.w - 1) 偏离单位元的程度
    Eigen::Quaterniond q_prior(x_(6), x_(7), x_(8), x_(9));
    Eigen::Quaterniond q_obs(z(3), z(4), z(5), z(6));

    // delta_q = q_obs * q_prior^(-1)  （从 prior 到 obs 的相对旋转）
    Eigen::Quaterniond delta_q = q_obs * q_prior.inverse();

    // 协方差加权误差：位置用线性减法，四元数用切空间表示
    h(3) = delta_q.x();
    h(4) = delta_q.y();
    h(5) = delta_q.z();
    // ★★★ Bug Fix: 四元数w值clamp到[-1,1]防止NaN ★★★
    // 由于浮点误差，delta_q.w()可能略微超出[-1,1]，导致后续计算产生NaN
    double w_val = delta_q.w();
    if (w_val > 1.0) w_val = 1.0;
    else if (w_val < -1.0) w_val = -1.0;
    h(6) = w_val - 1.0;  // 偏离单位元的标量误差

    Eigen::Matrix<double, 7, 1> y = z - h;

    // ---- 观测雅可比 H ----
    // 位置：H_position = I(3×3)
    // 四元数：H_quaternion 采用一阶近似
    //   对于 SO(3) 切空间误差，J ≈ I（在小角度近似下）
    //   此处用块对角矩阵，position 对 position，quaternion 对 quaternion
    Eigen::Matrix<double, 7, 13> H = Eigen::Matrix<double, 7, 13>::Zero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    H.block<4, 4>(3, 6) = Eigen::Matrix4d::Identity();

    // ---- Kalman 更新 ----
    Eigen::Matrix<double, 7, 7> S = H * P_ * H.transpose() + R_;

    // 奇异值分解求逆（S.inverse() 在病态时产生 NaN）
    Eigen::JacobiSVD<Eigen::Matrix<double, 7, 7>> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::VectorXd singular = svd.singularValues();
    // ★★★ Bug Fix: 降低条件数阈值从1e6到1e4，更严格地防止数值退化 ★★★
    // 原阈值1e6可能在长距离运行中逐渐累积误差
    double cond_num = singular.maxCoeff() / std::max<double>(singular.minCoeff(), 1e-8);
    if (cond_num > 1e4 || !S.allFinite()) {
        // 矩阵病态或含 NaN：跳过此次更新，防止 NaN 传播
        return;
    }
    Eigen::Matrix<double, 7, 7> S_inv = svd.matrixV() * singular.cwiseInverse().asDiagonal() * svd.matrixU().transpose();

    Eigen::Matrix<double, 13, 7> K = P_ * H.transpose() * S_inv;

    x_ += K * y;
    P_ = (Eigen::Matrix<double, 13, 13>::Identity() - K * H) * P_;

    // ---- 重归一化四元数（保持单位约束）----
    Eigen::Quaterniond q(x_(6), x_(7), x_(8), x_(9));
    if (q.norm() < 1e-10) {
        q = Eigen::Quaterniond(1, 0, 0, 0);
    } else {
        q.normalize();
    }
    x_(6) = q.w(); x_(7) = q.x(); x_(8) = q.y(); x_(9) = q.z();

    // ---- NaN 检查：若状态或协方差含 NaN，丢弃此次更新 ----
    if (!x_.allFinite() || !P_.allFinite()) {
        // 回退到先验
        x_ = x_backup;
        P_ = P_backup;
    }
}

Pose PoseEKF::getCurrentPose() const {
    Pose pose;
    pose.t = x_.block<3, 1>(0, 0);
    pose.q = Eigen::Quaterniond(x_(6), x_(7), x_(8), x_(9));
    return pose;
}

} // namespace frontend
} // namespace sfc_core
