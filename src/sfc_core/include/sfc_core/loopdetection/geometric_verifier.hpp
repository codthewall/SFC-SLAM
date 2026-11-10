// 必须在最开头清除 CMake project() 注入的 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_LOOPDETECTION_GEOMETRIC_VERIFIER_HPP_
#define SFC_CORE_LOOPDETECTION_GEOMETRIC_VERIFIER_HPP_

#include "sfc_core/frontend/frame.hpp"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include <random>

namespace sfc_core {
namespace loopdetection {

// RANSAC-based geometric verification using Essential Matrix
//
// This implements the geometric verification method used in ORB-SLAM:
// 1. Use RANSAC to estimate Essential Matrix from matched 2D-2D correspondences
// 2. Check inlier ratio - if > threshold, the match is geometrically consistent
// 3. Recover relative pose from Essential Matrix

struct GeometricVerifierConfig {
    int ransac_iterations = 200;
    double threshold_px = 4.0;        // Pixel threshold for inlier check
    double min_inlier_ratio = 0.20;   // ★ 从 0.25 放宽到 0.20（Essential Matrix 回环降级需要更宽松）
    int min_inliers = 4;              // ★ 从 10 降为 4（回环描述子匹配数少，门槛过高会导致所有候选失败）
    bool use_fundamental = false;      // Use Fundamental instead of Essential

    // ICP 专用参数（比 Essential Matrix 更宽松，因为是 3D-3D 直接对齐）
    // 注意：3D-3D ICP 只需要 3 个对应即可求解（比 2D-2D 的 5/8 点更稳定）
    // ★★★ 提高质量门槛：从 2 改为 4 ★★★
    // 只有 2 个内点时可靠性极低（统计上噪声容易主导），提高到 4 个内点
    // 同时要求内点率 >= 0.20，确保至少有 20% 的匹配是内点
    int min_icp_inliers = 4;            // 从 2 提高到 4（可靠性要求）
    double min_icp_inlier_ratio = 0.20; // 从 0.15 提高到 0.20（至少 20% 内点率）
    // ★★★ ICP 重投影误差阈值 ★★★
    // 室内场景特征点尺度较小，适度宽松的阈值有助于提高内点数量
    double icp_reproj_thresh_px = 20.0;  // 放宽至 20px
    // ★★★ 退化检测阈值 ★★★
    // 室内场景高内点率是正常的，不应轻易判定为退化
    double max_icp_inlier_ratio = 0.98;  // 几乎禁用退化检测

    // 回环场景下 Essential Matrix 降级方案专用参数
    // 回环描述子匹配数通常很少（7-14 个），需要比普通几何验证更宽松的阈值
    int min_essential_inliers = 3;         // 从 4 降低到 3（室内场景有效匹配对经常只有 3 个）
    double min_essential_inlier_ratio = 0.40;

    // 3D 深度过滤范围：超出范围的深度点不参与 ICP 计算（影响 ICP 有效匹配数）。
    double min_depth = 0.1;   // 最小有效深度（米），与 Tracking::config_.min_depth=0.1 一致
    double max_depth = 12.0;   // 最大有效深度（米），与 Tracking::config_.max_depth=12.0 一致
                                  // ★★★ P0-1: 修复深度阈值不一致 ★★★
                                  // 原值 0.05-30.0 与 Tracking 使用的 0.1-12.0 不一致，
                                  // 导致回环 ICP 使用错误的 3D 点集合验证

    // L10：诊断输出开关。本文件原有 45 处无条件 std::cerr，而回环检测在正常
    // 序列上"没找到回环"是常态 —— 每帧把 ICP 内点数、Sim3 中间结果全打到 stderr，
    // 真正的失败信息会被淹没（R-017/R-021）。默认关闭，由 LoopDetectorConfig::verbose
    // 一路注入。错误级日志（参数非法、容器越界这类"程序确有 bug"的信号）保持无条件。
    bool verbose = false;
};

class GeometricVerifier {
public:
    explicit GeometricVerifier(const GeometricVerifierConfig& config = GeometricVerifierConfig())
        : config_(config) {}

    int verifyWithEssential(const std::vector<cv::KeyPoint>& kp1,
                           const std::vector<cv::KeyPoint>& kp2,
                           const std::vector<cv::DMatch>& matches,
                           const cv::Mat& K,
                           Eigen::Matrix4d* out_relative_pose = nullptr);

    // ★ Homography 验证 - 用于平面场景或纯旋转场景
    // 当场景中存在平面时，Homography 比 Essential Matrix 更准确
    int verifyWithHomography(const std::vector<cv::KeyPoint>& kp1,
                           const std::vector<cv::KeyPoint>& kp2,
                           const std::vector<cv::DMatch>& matches,
                           Eigen::Matrix4d* out_relative_pose = nullptr);

    // ★ HE (Homography + Essential) 融合验证
    // 自动选择更好的方法：平面场景用 Homography，非平面用 Essential
    int verifyWithHE(const std::vector<cv::KeyPoint>& kp1,
                   const std::vector<cv::KeyPoint>& kp2,
                   const std::vector<cv::DMatch>& matches,
                   const cv::Mat& K,
                   Eigen::Matrix4d* out_relative_pose = nullptr);

    // ★ Sim(3) 尺度估计 - 从 Essential Matrix 估计真实尺度
    // 使用三角测量点的深度来估计相对位姿的真实尺度
    double estimateScaleSim3(const std::vector<cv::KeyPoint>& kp1,
                           const std::vector<cv::KeyPoint>& kp2,
                           const std::vector<cv::DMatch>& matches,
                           const cv::Mat& K,
                           const Eigen::Matrix4d& T_cw,
                           const Eigen::Matrix4d& T_matched_w,
                           const Eigen::Vector3d& baseline_direction);

    // ★ RGB-D 专用：3D-3D ICP 对齐（带 RANSAC），输出带真实尺度的相对位姿
    // pts3d1: 当前帧在相机系下的 3D 点（已深度反投）
    // pts3d2: 匹配帧在相机系下的 3D 点
    // matches: queryIdx 对应 pts3d1，trainIdx 对应 pts3d2
    // 返回内点数，out_relative_pose 输出 T_c1←c2（即从帧2相机系到帧1相机系）
    int verifyWithICP3D(const std::vector<Eigen::Vector3d>& pts3d1,
                        const std::vector<Eigen::Vector3d>& pts3d2,
                        const std::vector<cv::DMatch>& matches,
                        const cv::Mat& K,
                        Eigen::Matrix4d* out_relative_pose = nullptr);

    int verifyWithFundamental(const std::vector<cv::KeyPoint>& kp1,
                            const std::vector<cv::KeyPoint>& kp2,
                            const std::vector<cv::DMatch>& matches,
                            Eigen::Matrix4d* out_relative_pose = nullptr);

    cv::Mat getEssentialMatrix() const { return E_; }
    std::vector<bool> getInlierMask() const { return inlier_mask_; }
    const GeometricVerifierConfig& getConfig() const { return config_; }

private:
    GeometricVerifierConfig config_;
    cv::Mat E_;
    std::vector<bool> inlier_mask_;

    void matchesToPoints(const std::vector<cv::KeyPoint>& kp1,
                        const std::vector<cv::KeyPoint>& kp2,
                        const std::vector<cv::DMatch>& matches,
                        std::vector<cv::Point2f>& pts1,
                        std::vector<cv::Point2f>& pts2);
};

// =============================================================================
// Implementation
// =============================================================================

inline void GeometricVerifier::matchesToPoints(const std::vector<cv::KeyPoint>& kp1,
                                             const std::vector<cv::KeyPoint>& kp2,
                                             const std::vector<cv::DMatch>& matches,
                                             std::vector<cv::Point2f>& pts1,
                                             std::vector<cv::Point2f>& pts2) {
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    for (const auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }
}

inline int GeometricVerifier::verifyWithEssential(const std::vector<cv::KeyPoint>& kp1,
                                                 const std::vector<cv::KeyPoint>& kp2,
                                                 const std::vector<cv::DMatch>& matches,
                                                 const cv::Mat& K,
                                                 Eigen::Matrix4d* out_relative_pose) {
    if (matches.size() < static_cast<size_t>(config_.min_inliers)) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] Not enough matches: " << matches.size()
                  << " < min_inliers=" << config_.min_inliers << std::endl; }
        return -1;
    }

    std::vector<cv::Point2f> pts1, pts2;
    matchesToPoints(kp1, kp2, matches, pts1, pts2);

    // 使用 OpenCV 的 findEssentialMat 进行鲁棒的 Essential Matrix 估计
    cv::Mat inlier_mask;
    E_ = cv::findEssentialMat(pts1, pts2, K,
                               cv::RANSAC, 0.999, config_.threshold_px,
                               inlier_mask);

    if (E_.empty()) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] Failed to compute Essential Matrix" << std::endl; }
        return -1;
    }

    // 统计内点
    int inlier_count = 0;
    inlier_mask_.assign(matches.size(), false);
    for (size_t i = 0; i < matches.size(); i++) {
        if (inlier_mask.at<uchar>(i) > 0) {
            inlier_mask_[i] = true;
            inlier_count++;
        }
    }

    double inlier_ratio = static_cast<double>(inlier_count) / matches.size();

    { if (config_.verbose) std::cerr << "[GeoVerifier] Essential: inliers=" << inlier_count
              << "/" << matches.size() << " (" << std::fixed << std::setprecision(1)
              << inlier_ratio * 100 << "%)" << std::endl; }

    // 检查内点数量和内点率
    if (inlier_count < config_.min_inliers) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] FAIL: inliers=" << inlier_count
                  << " < min_inliers=" << config_.min_inliers << std::endl; }
        return -1;
    }

    if (inlier_ratio < config_.min_inlier_ratio) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] FAIL: ratio=" << std::fixed << std::setprecision(2)
                  << inlier_ratio << " < min_ratio=" << config_.min_inlier_ratio << std::endl; }
        return -1;
    }

    // 从 Essential Matrix 恢复位姿
    if (out_relative_pose) {
        cv::Mat R, t;
        std::vector<cv::Point2f> inlier_pts1, inlier_pts2;
        for (size_t i = 0; i < matches.size(); i++) {
            if (inlier_mask_[i]) {
                inlier_pts1.push_back(pts1[i]);
                inlier_pts2.push_back(pts2[i]);
            }
        }

        // 安全检查：recoverPose 要求 E 是 3x3，行列式 > 0
        if (E_.rows != 3 || E_.cols != 3 || E_.channels() != 1) {
            { if (config_.verbose) std::cerr << "[GeoVerifier] Essential Matrix 退化（尺寸="
                      << E_.rows << "x" << E_.cols
                      << "），跳过 recoverPose" << std::endl; }
            out_relative_pose->setIdentity();
            return inlier_count;
        }
        if (static_cast<int>(inlier_pts1.size()) < 5) {
            std::cerr << "[GeoVerifier] 内点数不足（" << inlier_pts1.size()
                      << " < 5），跳过 recoverPose" << std::endl;
            out_relative_pose->setIdentity();
            return inlier_count;
        }

        int pose_ok = cv::recoverPose(E_, inlier_pts1, inlier_pts2, K, R, t);

        if (pose_ok > 0) {
            Eigen::Matrix3d R_eigen;
            R_eigen << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                       R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
                       R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
            Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2));

            out_relative_pose->setIdentity();
            out_relative_pose->block<3, 3>(0, 0) = R_eigen;
            out_relative_pose->block<3, 1>(0, 3) = t_eigen;
            { if (config_.verbose) std::cerr << "[GeoVerifier] Relative pose: |t|=" << t_eigen.norm() << " m" << std::endl; }
        } else {
            // ★ 优化 5：recoverPose 退化（返回 0）时，用 SVD 分解回退获取平移方向
            // 避免完全丢失几何约束的平移信息，替代原来直接返回单位阵
            { if (config_.verbose) std::cerr << "[GeoVerifier] recoverPose 退化（返回 " << pose_ok
                      << "），使用 SVD 分解回退获取平移方向..." << std::endl; }
            cv::SVD svd(E_);
            cv::Mat U = svd.u;
            cv::Mat VT = svd.vt;
            Eigen::Vector3d t_eigen(
                U.at<double>(0, 2),
                U.at<double>(1, 2),
                U.at<double>(2, 2));
            double t_norm = std::sqrt(t_eigen.squaredNorm());
            if (t_norm > 1e-10) t_eigen /= t_norm;
            Eigen::Matrix3d R_eigen;
            if (R.type() == CV_64F) {
                R_eigen << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
                           R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
                           R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
            } else {
                R_eigen << R.at<float>(0, 0), R.at<float>(0, 1), R.at<float>(0, 2),
                           R.at<float>(1, 0), R.at<float>(1, 1), R.at<float>(1, 2),
                           R.at<float>(2, 0), R.at<float>(2, 1), R.at<float>(2, 2);
                if (R_eigen.determinant() < 0) R_eigen *= -1.0;
            }
            out_relative_pose->setIdentity();
            out_relative_pose->block<3, 3>(0, 0) = R_eigen;
            out_relative_pose->block<3, 1>(0, 3) = t_eigen;
            { if (config_.verbose) std::cerr << "[GeoVerifier] SVD 回退平移方向: |t|=" << t_eigen.norm()
                      << "（已归一化，仅方向可靠）" << std::endl; }
        }
    }

    { if (config_.verbose) std::cerr << "[GeoVerifier] PASS: Geometric verification successful!" << std::endl; }
    return inlier_count;
}

// ================================================================
// 辅助函数：计算给定 R,t 的平均重投影误差（用于翻转判决）
// ================================================================
static double computeMeanReprojError(
        const Eigen::Matrix3d& R_in,
        const Eigen::Vector3d& t_in,
        const std::vector<Eigen::Vector3d>& pts3d1,
        const std::vector<Eigen::Vector3d>& pts3d2,
        const std::vector<cv::DMatch>& vmatches,
        double fx, double fy, double cx, double cy,
        double inlier_thresh_sq) {
    int N = static_cast<int>(vmatches.size());
    if (N == 0) return 1e9;
    double total_err = 0.0;
    int cnt = 0;
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d p2_cam = R_in * pts3d2[vmatches[i].trainIdx] + t_in;
        Eigen::Vector3d p1_obs = pts3d1[vmatches[i].queryIdx];
        if (p2_cam.z() < 0.05 || p1_obs.z() < 0.05) continue;
        double invz1 = 1.0 / p1_obs.z();
        double invz2 = 1.0 / p2_cam.z();
        if (!std::isfinite(invz1) || !std::isfinite(invz2)) continue;
        double u_obs = fx * p1_obs.x() * invz1 + cx;
        double v_obs = fy * p1_obs.y() * invz1 + cy;
        double u_proj = fx * p2_cam.x() * invz2 + cx;
        double v_proj = fy * p2_cam.y() * invz2 + cy;
        double err_sq = (u_obs - u_proj) * (u_obs - u_proj) +
                        (v_obs - v_proj) * (v_obs - v_proj);
        total_err += err_sq;
        cnt++;
    }
    return (cnt > 0) ? (total_err / cnt) : 1e9;
}

// ================================================================
// 辅助函数：用 SVD 计算 3-point 对应的 R,t
// ================================================================
static void svdFrom3Points(
        const std::vector<Eigen::Vector3d>& p1_set,
        const std::vector<Eigen::Vector3d>& p2_set,
        Eigen::Matrix3d* out_R,
        Eigen::Vector3d* out_t) {
    Eigen::Vector3d c1 = (p1_set[0] + p1_set[1] + p1_set[2]) / 3.0;
    Eigen::Vector3d c2 = (p2_set[0] + p2_set[1] + p2_set[2]) / 3.0;
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; ++i) {
        Eigen::Vector3d d1 = p1_set[i] - c1;
        Eigen::Vector3d d2 = p2_set[i] - c2;
        H += d2 * d1.transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d Rv = svd.matrixV() * svd.matrixU().transpose();

    // ★★★ 修复：行列式检查和 SVD 正交化 ★★★
    // 如果 det(R) < 0，说明得到的是反射而非旋转，需要翻转修正
    if (Rv.determinant() < 0) {
        // 翻转 V 的最后一列（对应最小奇异值），重新计算
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1;
        Rv = V * svd.matrixU().transpose();
    }
    // SVD 正交化作为最后防线
    {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd_fix(Rv,
            Eigen::ComputeFullU | Eigen::ComputeFullV);
        Rv = svd_fix.matrixU() * svd_fix.matrixV().transpose();
        if (Rv.determinant() < 0) Rv.col(2) *= -1;
    }
    *out_R = Rv;
    *out_t = c1 - Rv * c2;
}

// ================================================================
// 辅助函数：统计内点数（给定 R,t）
// ================================================================
static int countInliersFromRT(
        const Eigen::Matrix3d& R_in,
        const Eigen::Vector3d& t_in,
        const std::vector<Eigen::Vector3d>& pts3d1,
        const std::vector<Eigen::Vector3d>& pts3d2,
        const std::vector<cv::DMatch>& vmatches,
        double fx, double fy, double cx, double cy,
        double inlier_thresh_sq) {
    int N = static_cast<int>(vmatches.size());
    int cnt = 0;
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d p2_cam = R_in * pts3d2[vmatches[i].trainIdx] + t_in;
        Eigen::Vector3d p1_obs = pts3d1[vmatches[i].queryIdx];
        if (p2_cam.z() < 1e-6 || p1_obs.z() < 1e-6) continue;
        double u_obs = fx * p1_obs.x() / p1_obs.z() + cx;
        double v_obs = fy * p1_obs.y() / p1_obs.z() + cy;
        double u_proj = fx * p2_cam.x() / p2_cam.z() + cx;
        double v_proj = fy * p2_cam.y() / p2_cam.z() + cy;
        double err_sq = (u_obs - u_proj) * (u_obs - u_proj) +
                        (v_obs - v_proj) * (v_obs - v_proj);
        if (err_sq < inlier_thresh_sq) cnt++;
    }
    return cnt;
}

// ================================================================
// 辅助函数：穷举所有 C(N,3) 组合，找内点最多的 SVD 解
// ================================================================
static std::pair<int, std::pair<Eigen::Matrix3d, Eigen::Vector3d>>
enumerate3PointCombinations(
        int N,
        const std::vector<Eigen::Vector3d>& pts3d1,
        const std::vector<Eigen::Vector3d>& pts3d2,
        const std::vector<cv::DMatch>& vmatches,
        double fx, double fy, double cx, double cy,
        double inlier_thresh_sq) {
    int best_inliers = 0;
    Eigen::Matrix3d best_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d best_t = Eigen::Vector3d::Zero();

    // 遍历所有 C(N,3) 组合
    for (int i0 = 0; i0 < N - 2; ++i0) {
        for (int i1 = i0 + 1; i1 < N - 1; ++i1) {
            for (int i2 = i1 + 1; i2 < N; ++i2) {
                std::vector<Eigen::Vector3d> p1_s = {
                    pts3d1[vmatches[i0].queryIdx],
                    pts3d1[vmatches[i1].queryIdx],
                    pts3d1[vmatches[i2].queryIdx]
                };
                std::vector<Eigen::Vector3d> p2_s = {
                    pts3d2[vmatches[i0].trainIdx],
                    pts3d2[vmatches[i1].trainIdx],
                    pts3d2[vmatches[i2].trainIdx]
                };
                Eigen::Matrix3d R3;
                Eigen::Vector3d t3;
                svdFrom3Points(p1_s, p2_s, &R3, &t3);

                // ★★★ 修复：正确的翻转判决（基于行列式，而非只翻转 z 轴）★★★
                // SVD 得到的旋转矩阵可能行列式为负（Horn 1987 方法中第二优解）
                // 正确的翻转应该是：翻转 V 的最后一列，重新计算 R = V * U^T
                // 但更简单的方法是：检测 det(R) < 0 并翻转列向量来修正
                if (R3.determinant() < 0) {
                    // 翻转第三列（对应最小奇异值对应的方向）
                    R3.col(2) *= -1.0;
                }
                // 再次验证并做 SVD 正交化作为最后防线
                {
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd_fix(R3,
                        Eigen::ComputeFullU | Eigen::ComputeFullV);
                    R3 = svd_fix.matrixU() * svd_fix.matrixV().transpose();
                    if (R3.determinant() < 0) R3.col(2) *= -1.0;
                }

                // 对比翻转前后的重投影误差，选择更好的
                double err_before = computeMeanReprojError(
                    R3, t3, pts3d1, pts3d2, vmatches, fx, fy, cx, cy, inlier_thresh_sq);
                // 翻转的版本：行列式取反
                Eigen::Matrix3d R3_flip = R3;
                R3_flip.col(2) *= -1.0;
                double err_after = computeMeanReprojError(
                    R3_flip, t3, pts3d1, pts3d2, vmatches, fx, fy, cx, cy, inlier_thresh_sq);
                if (err_after < err_before * 0.9) {  // 翻转后误差明显更小（10%阈值）
                    R3 = R3_flip;
                }

                int n_inliers = countInliersFromRT(
                    R3, t3, pts3d1, pts3d2, vmatches, fx, fy, cx, cy, inlier_thresh_sq);
                if (n_inliers > best_inliers) {
                    best_inliers = n_inliers;
                    best_R = R3;
                    best_t = t3;
                }
            }
        }
    }
    return {best_inliers, {best_R, best_t}};
}

// ================================================================
// 辅助函数：全部点 SVD（使用全部 N 个点）
// ================================================================
static std::pair<int, std::pair<Eigen::Matrix3d, Eigen::Vector3d>>
allPointsSVD(
        int N,
        const std::vector<Eigen::Vector3d>& pts3d1,
        const std::vector<Eigen::Vector3d>& pts3d2,
        const std::vector<cv::DMatch>& vmatches,
        double fx, double fy, double cx, double cy,
        double inlier_thresh_sq) {
    Eigen::Vector3d centroid1 = Eigen::Vector3d::Zero();
    Eigen::Vector3d centroid2 = Eigen::Vector3d::Zero();
    for (int i = 0; i < N; ++i) {
        centroid1 += pts3d1[vmatches[i].queryIdx];
        centroid2 += pts3d2[vmatches[i].trainIdx];
    }
    centroid1 /= N;
    centroid2 /= N;

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d d1 = pts3d1[vmatches[i].queryIdx] - centroid1;
        Eigen::Vector3d d2 = pts3d2[vmatches[i].trainIdx] - centroid2;
        H += d2 * d1.transpose();
    }
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d R_all = svd.matrixV() * svd.matrixU().transpose();
    if (R_all.determinant() < 0) {
        Eigen::Matrix3d V = svd.matrixV();
        V.col(2) *= -1;
        R_all = V * svd.matrixU().transpose();
    }
    // ★★★ 添加：SVD 正交化验证 ★★★
    {
        double det = R_all.determinant();
        double ortho_err = (R_all * R_all.transpose() - Eigen::Matrix3d::Identity()).norm();
        if (std::abs(det - 1.0) > 1e-6 || ortho_err > 1e-6) {
            Eigen::JacobiSVD<Eigen::Matrix3d> svd_fix(R_all,
                Eigen::ComputeFullU | Eigen::ComputeFullV);
            R_all = svd_fix.matrixU() * svd_fix.matrixV().transpose();
            if (R_all.determinant() < 0) R_all.col(2) *= -1.0;
        }
    }
    Eigen::Vector3d t_all = centroid1 - R_all * centroid2;

    // ★★★ 改进翻转判决：先检查行列式，再对比误差 ★★★
    // 如果 det(R) < 0，说明旋转矩阵需要翻转
    bool need_flip = (R_all.determinant() < 0);
    double err_before = 1e9, err_after = 1e9;

    if (need_flip) {
        R_all.col(2) *= -1.0;
    }

    // 计算翻转前后的重投影误差
    err_before = computeMeanReprojError(
        R_all, t_all, pts3d1, pts3d2, vmatches, fx, fy, cx, cy, inlier_thresh_sq);
    Eigen::Matrix3d R_all_flip = R_all;
    R_all_flip.col(2) *= -1.0;
    err_after = computeMeanReprojError(
        R_all_flip, t_all, pts3d1, pts3d2, vmatches, fx, fy, cx, cy, inlier_thresh_sq);

    // 只有当翻转后误差明显更小时才翻转（10%阈值，避免数值抖动）
    if (err_after < err_before * 0.9) {
        R_all = R_all_flip;
    }

    int n_inliers = countInliersFromRT(
        R_all, t_all, pts3d1, pts3d2, vmatches, fx, fy, cx, cy, inlier_thresh_sq);
    return {n_inliers, {R_all, t_all}};
}

// ================================================================
// 主函数：verifyWithICP3D（重写版）
// ================================================================
inline int GeometricVerifier::verifyWithICP3D(
        const std::vector<Eigen::Vector3d>& pts3d1,
        const std::vector<Eigen::Vector3d>& pts3d2,
        const std::vector<cv::DMatch>& matches,
        const cv::Mat& K,
        Eigen::Matrix4d* out_relative_pose) {
    // ============================================================
    // RGB-D 专用 3D-3D ICP：对齐两帧的深度点云，带真实尺度
    //
    // ★ 优化 2+3 重写策略：
    //   1. 小样本（N ≤ 8）：穷举所有 C(N,3) 组合，避免随机 RANSAC 的不稳定性
    //   2. 大样本（N > 8）：全部点 SVD + 随机 RANSAC 精化
    //   3. 翻转判决：对比翻转前后平均误差，而非固定比例阈值
    // ============================================================

    if (matches.size() < 4) {
        { if (config_.verbose) std::cerr << "[GeoVerifier ICP] Not enough matches: " << matches.size() << " < 4" << std::endl; }
        return -1;
    }

    // 过滤掉无效 3D 点
    std::vector<cv::DMatch> valid_matches;
    valid_matches.reserve(matches.size());
    for (const auto& m : matches) {
        int i1 = m.queryIdx;
        int i2 = m.trainIdx;
        if (i1 >= static_cast<int>(pts3d1.size()) ||
            i2 >= static_cast<int>(pts3d2.size())) continue;
        if (pts3d1[i1].norm() < 1e-6 || pts3d2[i2].norm() < 1e-6) continue;
        if (pts3d1[i1].z() < config_.min_depth || pts3d1[i1].z() > config_.max_depth) continue;
        if (pts3d2[i2].z() < config_.min_depth || pts3d2[i2].z() > config_.max_depth) continue;
        valid_matches.push_back(m);
    }

    if (static_cast<int>(valid_matches.size()) < 3) {
        { if (config_.verbose) std::cerr << "[GeoVerifier ICP] Not enough valid 3D matches: "
                  << valid_matches.size() << " < 3" << std::endl; }
        return -1;
    }

    int N = static_cast<int>(valid_matches.size());
    { if (config_.verbose) std::cerr << "[GeoVerifier ICP] Valid 3D pairs: " << N
              << " / " << matches.size() << std::endl; }

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    const double inlier_thresh_sq = config_.icp_reproj_thresh_px * config_.icp_reproj_thresh_px;

    // ============================================================
    // ★★★ 核心修复：替换随机 RANSAC 为穷举/全局 SVD ★★★
    // ============================================================
    int best_inliers = 0;
    Eigen::Matrix3d best_R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d best_t = Eigen::Vector3d::Zero();

    if (N <= 8) {
        // ---- 小样本：穷举所有 C(N,3) 组合 ----
        // C(5,3)=10，C(6,3)=20，C(7,3)=35，C(8,3)=56，完全可穷举
        { if (config_.verbose) std::cerr << "[GeoVerifier ICP] ★ 小样本（N=" << N
                  << "），穷举 C(" << N << ",3)=" << (N*(N-1)*(N-2)/6)
                  << " 个组合..." << std::endl; }
        auto result = enumerate3PointCombinations(
            N, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);
        best_inliers = result.first;
        best_R = result.second.first;
        best_t = result.second.second;

        // 如果穷举的内点不足，再用全部点 SVD 补充
        if (best_inliers < config_.min_icp_inliers) {
            auto result_all = allPointsSVD(
                N, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);
            if (result_all.first > best_inliers) {
                best_inliers = result_all.first;
                best_R = result_all.second.first;
                best_t = result_all.second.second;
                { if (config_.verbose) std::cerr << "[GeoVerifier ICP] 全部点 SVD 优于穷举，采用: inliers="
                          << best_inliers << std::endl; }
            }
        }
    } else {
        // ---- 大样本：全部点 SVD + 随机 RANSAC 精化 ----
        { if (config_.verbose) std::cerr << "[GeoVerifier ICP] ★ 大样本（N=" << N
                  << "），全部点 SVD + RANSAC 精化..." << std::endl; }

        // 步骤 1：全部点 SVD 作为初始估计
        auto result_all = allPointsSVD(
            N, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);
        best_inliers = result_all.first;
        best_R = result_all.second.first;
        best_t = result_all.second.second;

        // 步骤 2：用全部点 SVD 的内点做 RANSAC 精化
        // 从内点中随机采样，避免随机采到外点
        if (best_inliers >= config_.min_icp_inliers) {
            std::vector<int> inlier_idx;
            for (int i = 0; i < N; ++i) {
                Eigen::Vector3d p2_cam = best_R * pts3d2[valid_matches[i].trainIdx] + best_t;
                Eigen::Vector3d p1_obs = pts3d1[valid_matches[i].queryIdx];
                if (p2_cam.z() < 1e-6 || p1_obs.z() < 1e-6) continue;
                double u_obs = fx * p1_obs.x() / p1_obs.z() + cx;
                double v_obs = fy * p1_obs.y() / p1_obs.z() + cy;
                double u_proj = fx * p2_cam.x() / p2_cam.z() + cx;
                double v_proj = fy * p2_cam.y() / p2_cam.z() + cy;
                double err_sq = (u_obs - u_proj)*(u_obs - u_proj) +
                                (v_obs - v_proj)*(v_obs - v_proj);
                if (err_sq < inlier_thresh_sq) inlier_idx.push_back(i);
            }

            if (static_cast<int>(inlier_idx.size()) >= 3) {
                std::mt19937 rng(static_cast<unsigned>(N * 7919 + valid_matches.size() * 3571));
                const int refine_iters = 500;
                for (int iter = 0; iter < refine_iters; ++iter) {
                    const int max_idx = static_cast<int>(inlier_idx.size()) - 1;
                    if (max_idx < 0) break;
                    std::uniform_int_distribution<int> dist(0, max_idx);
                    int i0 = dist(rng), i1 = dist(rng), i2 = dist(rng);
                    while (i1 == i0) i1 = dist(rng);
                    while (i2 == i0 || i2 == i1) i2 = dist(rng);

                    std::vector<Eigen::Vector3d> p1_s = {
                        pts3d1[valid_matches[inlier_idx[i0]].queryIdx],
                        pts3d1[valid_matches[inlier_idx[i1]].queryIdx],
                        pts3d1[valid_matches[inlier_idx[i2]].queryIdx]
                    };
                    std::vector<Eigen::Vector3d> p2_s = {
                        pts3d2[valid_matches[inlier_idx[i0]].trainIdx],
                        pts3d2[valid_matches[inlier_idx[i1]].trainIdx],
                        pts3d2[valid_matches[inlier_idx[i2]].trainIdx]
                    };
                    Eigen::Matrix3d Rs, R_check;
                    Eigen::Vector3d ts;
                    svdFrom3Points(p1_s, p2_s, &Rs, &ts);

                    // ★★★ 修复翻转判决：先检查行列式，再对比误差 ★★★
                    // svdFrom3Points 已经做了行列式检查，这里再验证一次
                    if (Rs.determinant() < 0) {
                        Rs.col(2) *= -1.0;
                    }
                    // 计算翻转前后的重投影误差
                    double err_before = computeMeanReprojError(
                        Rs, ts, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);
                    Eigen::Matrix3d Rs_flip = Rs;
                    Rs_flip.col(2) *= -1.0;
                    double err_after = computeMeanReprojError(
                        Rs_flip, ts, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);
                    // 只有当翻转后误差明显更小时才翻转
                    R_check = (err_after < err_before * 0.9) ? Rs_flip : Rs;

                    int n_inliers = countInliersFromRT(
                        R_check, ts, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);
                    if (n_inliers > best_inliers) {
                        best_inliers = n_inliers;
                        best_R = R_check;
                        best_t = ts;
                    }
                }
            }
        }
    }

    // 最终验证
    if (!best_R.array().allFinite() || !best_t.array().allFinite()) {
        { if (config_.verbose) std::cerr << "[GeoVerifier ICP] FAIL: NaN/Inf in best_R or best_t" << std::endl; }
        return -1;
    }

    double inlier_ratio = static_cast<double>(best_inliers) / N;
    { if (config_.verbose) std::cerr << "[GeoVerifier ICP] inliers=" << best_inliers
              << "/" << N << " (" << std::fixed << std::setprecision(1)
              << inlier_ratio * 100 << "%)"
              << " |t|=" << best_t.norm() << " m" << std::endl; }

    if (inlier_ratio > config_.max_icp_inlier_ratio) {
        { if (config_.verbose) std::cerr << "[GeoVerifier ICP] FAIL: ratio=" << inlier_ratio
                  << " > max=" << config_.max_icp_inlier_ratio
                  << "（内点率过高，可能退化，拒绝）" << std::endl; }
        return -1;
    }

    if (best_inliers >= config_.min_icp_inliers &&
        inlier_ratio >= config_.min_icp_inlier_ratio) {
        if (out_relative_pose) {
            out_relative_pose->setIdentity();
            out_relative_pose->block<3, 3>(0, 0) = best_R;
            out_relative_pose->block<3, 1>(0, 3) = best_t;
        }
        { if (config_.verbose) std::cerr << "[GeoVerifier ICP] PASS: 内点=" << best_inliers
                  << " ratio=" << std::fixed << std::setprecision(2) << inlier_ratio
                  << " |t|=" << best_t.norm() << "m" << std::endl; }
        return best_inliers;
    }

    // ---- 迭代 SVD 精化回退（适用于仍不满足阈值的情况）----
    { if (config_.verbose) std::cerr << "[GeoVerifier ICP] 内点不足（inliers=" << best_inliers
              << " ratio=" << std::fixed << std::setprecision(2) << inlier_ratio
              << "），尝试迭代 SVD 精化回退..." << std::endl; }

    Eigen::Matrix3d best_cv_R = best_R;
    Eigen::Vector3d best_cv_t = best_t;
    int best_cv_inliers = best_inliers;

    // ★★★ 优化：迭代 SVD 精化，使用更大的迭代次数和更宽松的终止条件 ★★★
    const int max_refine_iters = 150;  // 从 50 增加到 150（更多迭代=更多机会找到内点）
    for (int refine_iter = 0; refine_iter < max_refine_iters; ++refine_iter) {
        std::vector<int> inlier_indices;
        for (int i = 0; i < N; ++i) {
            Eigen::Vector3d p2_cam = best_cv_R * pts3d2[valid_matches[i].trainIdx] + best_cv_t;
            Eigen::Vector3d p1_obs = pts3d1[valid_matches[i].queryIdx];
            if (p2_cam.z() < 1e-6 || p1_obs.z() < 1e-6) continue;
            double u_obs = fx * p1_obs.x() / p1_obs.z() + cx;
            double v_obs = fy * p1_obs.y() / p1_obs.z() + cy;
            double u_proj = fx * p2_cam.x() / p2_cam.z() + cx;
            double v_proj = fy * p2_cam.y() / p2_cam.z() + cy;
            double err_sq = (u_obs - u_proj)*(u_obs - u_proj) +
                            (v_obs - v_proj)*(v_obs - v_proj);
            if (err_sq < inlier_thresh_sq) inlier_indices.push_back(i);
        }

        int cv_inliers = static_cast<int>(inlier_indices.size());
        
        // ★★★ 优化：使用阈值提升作为终止条件，而非仅依赖内点数量不增加 ★★★
        // 当内点数量达到 min_icp_inliers 且比率满足阈值时即可接受
        if (cv_inliers >= config_.min_icp_inliers) {
            double cv_ratio_check = static_cast<double>(cv_inliers) / N;
            if (cv_ratio_check >= config_.min_icp_inlier_ratio) {
                // 验证翻转判决
                Eigen::Matrix3d R_test = best_cv_R;
                Eigen::Vector3d t_test = best_cv_t;
                
                // 计算当前解的翻转
                Eigen::Matrix3d R_flip = best_cv_R;
                R_flip.col(2) *= -1;
                Eigen::Vector3d t_flip = best_cv_t;
                
                int n_normal = countInliersFromRT(
                    R_test, t_test, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);
                int n_flip = countInliersFromRT(
                    R_flip, t_flip, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);
                
                if (n_flip > n_normal) {
                    best_cv_R = R_flip;
                    // 平移不需要翻转
                }
                best_cv_inliers = std::max(n_normal, n_flip);
                break;
            }
        }
        
        if (cv_inliers <= best_cv_inliers) break;

        best_cv_inliers = cv_inliers;

        // 用内点子集重新 SVD 求解
        Eigen::Vector3d refine_c1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d refine_c2 = Eigen::Vector3d::Zero();
        for (int idx : inlier_indices) {
            refine_c1 += pts3d1[valid_matches[idx].queryIdx];
            refine_c2 += pts3d2[valid_matches[idx].trainIdx];
        }
        refine_c1 /= inlier_indices.size();
        refine_c2 /= inlier_indices.size();

        std::vector<Eigen::Vector3d> rp1(inlier_indices.size()), rp2(inlier_indices.size());
        for (size_t i = 0; i < inlier_indices.size(); ++i) {
            rp1[i] = pts3d1[valid_matches[inlier_indices[i]].queryIdx] - refine_c1;
            rp2[i] = pts3d2[valid_matches[inlier_indices[i]].trainIdx] - refine_c2;
        }

        Eigen::Matrix3d H_refine = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < rp1.size(); ++i) H_refine += rp1[i] * rp2[i].transpose();

        Eigen::JacobiSVD<Eigen::Matrix3d> svd_refine(H_refine, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R_refine = svd_refine.matrixV() * svd_refine.matrixU().transpose();
        if (R_refine.determinant() < 0) {
            Eigen::Matrix3d Vref = svd_refine.matrixV();
            Vref.col(2) *= -1;
            R_refine = Vref * svd_refine.matrixU().transpose();
        }
        // ★★★ 添加：SVD 正交化验证 ★★★
        {
            double det = R_refine.determinant();
            double ortho_err = (R_refine * R_refine.transpose() - Eigen::Matrix3d::Identity()).norm();
            if (std::abs(det - 1.0) > 1e-6 || ortho_err > 1e-6) {
                Eigen::JacobiSVD<Eigen::Matrix3d> svd_fix(R_refine,
                    Eigen::ComputeFullU | Eigen::ComputeFullV);
                R_refine = svd_fix.matrixU() * svd_fix.matrixV().transpose();
                if (R_refine.determinant() < 0) R_refine.col(2) *= -1.0;
            }
        }
        Eigen::Vector3d t_refine = refine_c1 - R_refine * refine_c2;

        // ★★★ 改进翻转判决：行列式检查 + 误差对比 + SVD正交化 ★★★
        // 如果 det(R) < 0，先翻转
        if (R_refine.determinant() < 0) {
            R_refine.col(2) *= -1.0;
        }
        int n_normal = countInliersFromRT(
            R_refine, t_refine, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);
        Eigen::Matrix3d R_refine_flip = R_refine;
        R_refine_flip.col(2) *= -1.0;
        int n_flip = countInliersFromRT(
            R_refine_flip, t_refine, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);

        // SVD正交化翻转后的矩阵
        if (R_refine_flip.determinant() < 0) R_refine_flip.col(2) *= -1.0;

        // 选择内点更多的翻转状态（且翻转后误差明显更小时）
        double err_normal = 0, err_flip = 0;
        if (n_normal > n_flip) {
            best_cv_R = R_refine;
            best_cv_t = t_refine;
        } else if (n_flip > n_normal) {
            best_cv_R = R_refine_flip;
            best_cv_t = t_refine;
        } else {
            // 内点数相同，选择误差更小的
            err_normal = computeMeanReprojError(
                R_refine, t_refine, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);
            err_flip = computeMeanReprojError(
                R_refine_flip, t_refine, pts3d1, pts3d2, valid_matches, fx, fy, cx, cy, inlier_thresh_sq);
            if (err_flip < err_normal) {
                best_cv_R = R_refine_flip;
                best_cv_t = t_refine;
            } else {
                best_cv_R = R_refine;
                best_cv_t = t_refine;
            }
        }
    }

    int cv_inliers = 0;
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d p2_cam = best_cv_R * pts3d2[valid_matches[i].trainIdx] + best_cv_t;
        Eigen::Vector3d p1_obs = pts3d1[valid_matches[i].queryIdx];
        if (p2_cam.z() < 1e-6 || p1_obs.z() < 1e-6) continue;
        double u_obs = fx * p1_obs.x() / p1_obs.z() + cx;
        double v_obs = fy * p1_obs.y() / p1_obs.z() + cy;
        double u_proj = fx * p2_cam.x() / p2_cam.z() + cx;
        double v_proj = fy * p2_cam.y() / p2_cam.z() + cy;
        double err_sq = (u_obs - u_proj)*(u_obs - u_proj) +
                        (v_obs - v_proj)*(v_obs - v_proj);
        if (err_sq < inlier_thresh_sq) cv_inliers++;
    }

    double cv_ratio = static_cast<double>(cv_inliers) / N;
    { if (config_.verbose) std::cerr << "[GeoVerifier ICP] 迭代 SVD 精化: inliers=" << cv_inliers
              << "/" << N << " (" << std::fixed << std::setprecision(1) << cv_ratio * 100
              << "%)" << std::endl; }

    if (cv_inliers >= config_.min_icp_inliers &&
        cv_ratio >= config_.min_icp_inlier_ratio &&
        cv_ratio <= config_.max_icp_inlier_ratio &&
        best_cv_R.array().allFinite() && best_cv_t.array().allFinite()) {
        { if (config_.verbose) std::cerr << "[GeoVerifier ICP] 迭代 SVD 回退成功！" << std::endl; }
        if (out_relative_pose) {
            out_relative_pose->setIdentity();
            out_relative_pose->block<3, 3>(0, 0) = best_cv_R;
            out_relative_pose->block<3, 1>(0, 3) = best_cv_t;
        }
        return cv_inliers;
    }
    { if (config_.verbose) std::cerr << "[GeoVerifier ICP] 迭代 SVD 回退也失败" << std::endl; }
    return -1;
}

inline int GeometricVerifier::verifyWithFundamental(const std::vector<cv::KeyPoint>& kp1,
                                                    const std::vector<cv::KeyPoint>& kp2,
                                                    const std::vector<cv::DMatch>& matches,
                                                    Eigen::Matrix4d* /*out_relative_pose*/) {
    if (matches.size() < static_cast<size_t>(config_.min_inliers)) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] Not enough matches: " << matches.size()
                  << " < min_inliers=" << config_.min_inliers << std::endl; }
        return -1;
    }

    std::vector<cv::Point2f> pts1, pts2;
    matchesToPoints(kp1, kp2, matches, pts1, pts2);

    cv::Mat inlier_mask;
    cv::Mat F = cv::findFundamentalMat(pts1, pts2, cv::FM_RANSAC,
                                        config_.threshold_px, 0.999, inlier_mask);

    if (F.empty()) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] Failed to compute Fundamental Matrix" << std::endl; }
        return -1;
    }

    int inlier_count = 0;
    inlier_mask_.assign(matches.size(), false);
    for (size_t i = 0; i < matches.size(); i++) {
        if (inlier_mask.at<uchar>(i) > 0) {
            inlier_mask_[i] = true;
            inlier_count++;
        }
    }

    double inlier_ratio = static_cast<double>(inlier_count) / matches.size();

    { if (config_.verbose) std::cerr << "[GeoVerifier] Fundamental: inliers=" << inlier_count
              << "/" << matches.size() << " (" << std::fixed << std::setprecision(1)
              << inlier_ratio * 100 << "%)" << std::endl; }

    if (inlier_count < config_.min_inliers || inlier_ratio < config_.min_inlier_ratio) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] FAIL: Geometric verification failed" << std::endl; }
        return -1;
    }

    { if (config_.verbose) std::cerr << "[GeoVerifier] PASS: Geometric verification successful!" << std::endl; }
    return inlier_count;
}

inline double GeometricVerifier::estimateScaleSim3(const std::vector<cv::KeyPoint>& kp1,
                                                  const std::vector<cv::KeyPoint>& kp2,
                                                  const std::vector<cv::DMatch>& matches,
                                                  const cv::Mat& K,
                                                  const Eigen::Matrix4d& T_cw,
                                                  const Eigen::Matrix4d& T_matched_w,
                                                  const Eigen::Vector3d& baseline_direction) {
    // Sim(3) 尺度估计
    // 使用三角测量点的深度来估计相对位姿的真实尺度
    
    if (matches.size() < 8) return 1.0;
    
    std::vector<cv::Point2f> pts1, pts2;
    matchesToPoints(kp1, kp2, matches, pts1, pts2);
    
    // 获取本质矩阵
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, config_.threshold_px);
    if (E.empty()) return 1.0;
    
    // ★ 安全检查：recoverPose 要求 E 是 3x3，行列式 > 0（退化场景保护）
    if (E.rows != 3 || E.cols != 3 || E.channels() != 1) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] estimateScaleSim3: Essential Matrix 退化（尺寸="
                  << E.rows << "x" << E.cols << "），返回默认尺度 1.0" << std::endl; }
        return 1.0;
    }
    if (static_cast<int>(pts1.size()) < 5) return 1.0;

    // 恢复相对位姿
    cv::Mat R, t;
    cv::Mat inlier_mask_cv;  // 使用 cv::Mat 而非 vector<bool>
    cv::recoverPose(E, pts1, pts2, K, R, t, inlier_mask_cv);
    
    // 手动转换 cv::Mat 到 Eigen::Matrix3d
    Eigen::Matrix3d R_eigen;
    R_eigen << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
               R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
               R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    
    Eigen::Vector3d t_eigen(t.at<double>(0), t.at<double>(1), t.at<double>(2));
    
    // 归一化平移向量
    double t_norm = t_eigen.norm();
    if (t_norm < 1e-6) return 1.0;
    t_eigen.normalize();
    
    // 计算当前关键帧和匹配关键帧之间的相对位姿
    Eigen::Matrix4d T_rel = T_matched_w.inverse() * T_cw;
    Eigen::Vector3d t_actual = T_rel.block<3, 1>(0, 3);
    double actual_dist = t_actual.norm();
    
    // 估计尺度 = 实际距离 / 归一化距离
    double scale = (t_norm > 1e-6) ? (actual_dist / t_norm) : 1.0;
    
    // 限制尺度在合理范围 [0.5, 2.0]
    scale = std::max(0.5, std::min(2.0, scale));
    
    { if (config_.verbose) std::cerr << "[GeoVerifier] Sim(3) Scale estimation: raw_dist=" << t_norm 
              << " actual_dist=" << actual_dist << " scale=" << scale << std::endl; }
    
    return scale;
}

inline int GeometricVerifier::verifyWithHomography(const std::vector<cv::KeyPoint>& kp1,
                                                const std::vector<cv::KeyPoint>& kp2,
                                                const std::vector<cv::DMatch>& matches,
                                                Eigen::Matrix4d* out_relative_pose) {
    if (matches.size() < 4) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] Not enough matches for Homography: " << matches.size() << " < 4" << std::endl; }
        return -1;
    }
    
    std::vector<cv::Point2f> pts1, pts2;
    matchesToPoints(kp1, kp2, matches, pts1, pts2);
    
    // 计算 Homography
    cv::Mat inlier_mask;
    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, config_.threshold_px, inlier_mask);
    
    if (H.empty()) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] Failed to compute Homography Matrix" << std::endl; }
        return -1;
    }
    
    int inlier_count = 0;
    inlier_mask_.assign(matches.size(), false);
    for (size_t i = 0; i < matches.size(); i++) {
        if (inlier_mask.at<uchar>(i) > 0) {
            inlier_mask_[i] = true;
            inlier_count++;
        }
    }
    
    double inlier_ratio = static_cast<double>(inlier_count) / matches.size();
    { if (config_.verbose) std::cerr << "[GeoVerifier] Homography: inliers=" << inlier_count
              << "/" << matches.size() << " (" << std::fixed << std::setprecision(1)
              << inlier_ratio * 100 << "%)" << std::endl; }
    
    // Homography 的内点阈值可以更低（平面场景本来就容易满足）
    double homography_threshold = std::max(0.15, config_.min_inlier_ratio * 0.8);
    if (inlier_count < 4 || inlier_ratio < homography_threshold) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] Homography verification failed (ratio < " 
                  << homography_threshold << ")" << std::endl; }
        return -1;
    }
    
    if (out_relative_pose) {
        out_relative_pose->setIdentity();
        // 从 Homography 提取旋转（近似）
        // 注意：Homography 不能直接给出正确的相对位姿，这里用单位矩阵作为占位
        { if (config_.verbose) std::cerr << "[GeoVerifier] Homography recovered (rotation identity placeholder)" << std::endl; }
    }
    
    { if (config_.verbose) std::cerr << "[GeoVerifier] Homography verification PASSED!" << std::endl; }
    return inlier_count;
}

inline int GeometricVerifier::verifyWithHE(const std::vector<cv::KeyPoint>& kp1,
                                         const std::vector<cv::KeyPoint>& kp2,
                                         const std::vector<cv::DMatch>& matches,
                                         const cv::Mat& K,
                                         Eigen::Matrix4d* out_relative_pose) {
    // HE = Homography + Essential 融合验证
    // 自动选择更好的方法
    
    { if (config_.verbose) std::cerr << "\n[GeoVerifier] >>> HE (Homography + Essential) 融合验证" << std::endl; }
    
    // 先尝试 Homography
    int h_inliers = verifyWithHomography(kp1, kp2, matches, nullptr);
    
    // 再尝试 Essential
    int e_inliers = verifyWithEssential(kp1, kp2, matches, K, out_relative_pose);
    
    // 选择更好的方法
    if (h_inliers > 0 && e_inliers > 0) {
        double h_ratio = static_cast<double>(h_inliers) / matches.size();
        double e_ratio = static_cast<double>(e_inliers) / matches.size();
        
        { if (config_.verbose) std::cerr << "[GeoVerifier] HE Comparison: Homography=" << h_ratio*100 << "% Essential=" 
                  << e_ratio*100 << "%" << std::endl; }
        
        // 如果 Homography 内点率明显更高（说明是平面场景），使用 Essential 但保留其结果
        // 因为 Essential 仍然能提供正确的相对位姿
        if (e_ratio >= h_ratio * 0.8) {
            { if (config_.verbose) std::cerr << "[GeoVerifier] HE: Using Essential (non-planar scene)" << std::endl; }
            return e_inliers;
        } else {
            { if (config_.verbose) std::cerr << "[GeoVerifier] HE: Planar scene detected, Essential still valid" << std::endl; }
            // 即使是平面场景，Essential 仍能提供正确的相对位姿
            return e_inliers;
        }
    } else if (e_inliers > 0) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] HE: Essential passed" << std::endl; }
        return e_inliers;
    } else if (h_inliers > 0) {
        { if (config_.verbose) std::cerr << "[GeoVerifier] HE: Only Homography passed (pure rotation?)" << std::endl; }
        return h_inliers;
    }
    
    { if (config_.verbose) std::cerr << "[GeoVerifier] HE: All methods failed" << std::endl; }
    return -1;
}

}  // namespace loopdetection
}  // namespace sfc_core

#endif  // SFC_CORE_LOOPDETECTION_GEOMETRIC_VERIFIER_HPP_
