/**
 * SophusPoseAdapter implementation
 */

#ifdef sfc_core
#undef sfc_core
#endif

#include "sfc_core/frontend/sophus_pose_adapter.hpp"

#include <iostream>

namespace sfc_core {
namespace frontend {

// ================================================================
// SophusPose implementation
// ================================================================

SophusPose::SophusPose() : se3_(Sophus::SE3f()) {}

SophusPose::SophusPose(const Sophus::SE3f& se3) : se3_(se3) {}

SophusPose::SophusPose(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
    : se3_(R, t) {}

SophusPose::SophusPose(const Eigen::Quaternionf& q, const Eigen::Vector3f& t)
    : se3_(q, t) {}

SophusPose SophusPose::fromPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    return SophusPose(q.cast<float>(), t.cast<float>());
}

Eigen::Quaterniond SophusPose::toQuaterniond() const {
    return se3_.unit_quaternion().cast<double>();
}

Eigen::Vector3d SophusPose::toTranslationd() const {
    return se3_.translation().cast<double>();
}

SophusPose SophusPose::inverse() const {
    return SophusPose(se3_.inverse());
}

Eigen::Vector3f SophusPose::transformPoint(const Eigen::Vector3f& point) const {
    return se3_ * point;
}

Eigen::Vector3f SophusPose::transformPointInverse(const Eigen::Vector3f& point) const {
    return se3_.inverse() * point;
}

SophusPose SophusPose::operator*(const SophusPose& other) const {
    return SophusPose(se3_ * other.se3_);
}

void SophusPose::applyPerturbation(const Eigen::Vector3f& delta_translation,
                                   const Eigen::Vector3f& delta_rotation) {
    // 创建扰动（旋转用 so3，平移直接加）
    Sophus::SE3f::Tangent delta;
    delta << delta_rotation, delta_translation;
    se3_ = Sophus::SE3f::exp(delta) * se3_;
}

void SophusPose::setRotationMatrix(const Eigen::Matrix3f& R) {
    Eigen::Quaternionf q(R);
    se3_.setQuaternion(q);
}

void SophusPose::setQuaternion(const Eigen::Quaternionf& q) {
    se3_.setQuaternion(q);
}

void SophusPose::setTranslation(const Eigen::Vector3f& t) {
    se3_.translation() = t;
}

cv::Mat SophusPose::toCvMat4x4f() const {
    Eigen::Matrix4f T = se3_.matrix();
    cv::Mat mat(4, 4, CV_32F);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mat.at<float>(i, j) = T(i, j);
        }
    }
    return mat;
}

cv::Mat SophusPose::toCvMat4x4d() const {
    Eigen::Matrix4d T = se3_.matrix().cast<double>();
    cv::Mat mat(4, 4, CV_64F);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mat.at<double>(i, j) = T(i, j);
        }
    }
    return mat;
}

bool SophusPose::isValid() const {
    // 检查旋转矩阵的正交性
    Eigen::Matrix3f R = se3_.rotationMatrix();
    Eigen::Matrix3f I = R * R.transpose();
    float ortho_error = (I - Eigen::Matrix3f::Identity()).norm();
    return ortho_error < 1e-6;
}

// ================================================================
// PoseConverter implementation
// ================================================================

Sophus::SE3f PoseConverter::toSE3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    return Sophus::SE3f(q.cast<float>(), t.cast<float>());
}

void PoseConverter::toPose(const Sophus::SE3f& se3, Eigen::Quaterniond& q, Eigen::Vector3d& t) {
    q = se3.unit_quaternion().cast<double>();
    t = se3.translation().cast<double>();
}

Sophus::SE3f PoseConverter::toSE3(const cv::Mat& T) {
    if (T.empty() || T.rows != 4 || T.cols != 4) {
        return Sophus::SE3f();
    }
    
    Eigen::Matrix4f T_eigen;
    if (T.type() == CV_32F) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                T_eigen(i, j) = T.at<float>(i, j);
            }
        }
    } else {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                T_eigen(i, j) = static_cast<float>(T.at<double>(i, j));
            }
        }
    }
    
    return Sophus::SE3f(T_eigen.block<3, 3>(0, 0), T_eigen.block<3, 1>(0, 3));
}

cv::Mat PoseConverter::toCvMat(const Sophus::SE3f& se3) {
    Eigen::Matrix4f T = se3.matrix();
    cv::Mat mat(4, 4, CV_32F);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            mat.at<float>(i, j) = T(i, j);
        }
    }
    return mat;
}

Sophus::SE3f PoseConverter::computeRelativePose(const Sophus::SE3f& T1, const Sophus::SE3f& T2) {
    return T1.inverse() * T2;
}

} // namespace frontend
} // namespace sfc_core
