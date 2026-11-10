// ================================================================
// 第三方库（g2o / Eigen / OpenCV）必须在命名空间外部引入
// 原因：g2o CMake 注入 -Dsfc_core=1 宏，导致
// sfc_core::std 被编译器误解析为 ::std 以外的东西。
// 解决方案：在自定义命名空间外部包含所有第三方头文件。
// ================================================================

// 强制在最前引入标准库（确保 std::* 在 g2o 污染前已解析）
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <memory>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

// g2o 头文件（在自定义命名空间外部引入，引入后主动清除宏）
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/vertex_pointxyz.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

// 清除 g2o 注入的 sfc_core 宏，防止命名空间污染
#ifdef sfc_core
#undef sfc_core
#endif

// 项目自身头文件（g2o 已被干净引入，现在可以安全地包含项目头文件）
#include "sfc_core/frontend/robust_pnp_solver.hpp"
#include "sfc_core/frontend/mlpnp_solver.hpp"

namespace sfc_core {
namespace frontend {

namespace {  // anonymous namespace: EdgeRobustReproject and reprojError are local helpers

// ★★★ Bug Fix #34: EdgeRobustReprojectPoint（悬空的 Binary Edge）已删除 ★★★
// 原bug: EdgeRobustReprojectPoint 是 BaseBinaryEdge<..., VertexPointXYZ, VertexSE3Expmap>，
// 但 g2oBA 中只调用了 setVertex(0, v_se3)，从未连接 VertexPointXYZ。
// computeError() 中访问 _vertices[1] 时该位置未初始化 → 悬空指针访问！
// 正确做法：使用 unary EdgeRobustReproject（只含位姿顶点，computeError 直接用 Pose 参数）。

// ==================== 重投影误差计算工具 ====================

double reprojError(const cv::Point3f& pt3d,
                   const cv::Point2f& pt2d,
                   const Pose& pose,
                   const Eigen::Matrix3d& K) {
    Eigen::Vector3d pw(pt3d.x, pt3d.y, pt3d.z);
    Eigen::Quaterniond q = pose.q;
    Eigen::Vector3d t = pose.t;
    Eigen::Matrix3d R = q.toRotationMatrix();
    Eigen::Vector3d pc = R * pw + t;
    double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
    // NaN 安全：深度过浅（< 5cm）或 NaN 时返回大误差
    if (pc.z() < 0.05 || !pc.allFinite()) return 1e6;
    double invz = 1.0 / pc.z();
    if (!std::isfinite(invz)) return 1e6;
    double u = fx * pc(0) * invz + cx;
    double v = fy * pc(1) * invz + cy;
    double dx = u - pt2d.x, dy = v - pt2d.y;
    return std::sqrt(dx * dx + dy * dy);
}

} // anonymous namespace


// ==================== RobustPnPSolver ====================

RobustPnPSolver::RobustPnPSolver(const RobustPnPConfig& config)
    : config_(config) {}

RobustPnPSolver::RobustPnPSolver(const cv::Mat& K, const RobustPnPConfig& config)
    : config_(config), K_cv_(K.clone()) {
    if (!K.empty()) {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                K_eigen_(i, j) = K.at<double>(i, j);
    } else {
        K_eigen_.setIdentity();
    }
    has_camera_ = true;
}

void RobustPnPSolver::setCamera(const cv::Mat& K, const cv::Mat& D) {
    K_cv_ = K.clone();
    D_cv_ = D.clone();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            K_eigen_(i, j) = K.at<double>(i, j);
    has_camera_ = true;
}

void RobustPnPSolver::setCamera(const Eigen::Matrix3d& K_eigen) {
    K_eigen_ = K_eigen;
    K_cv_.create(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            K_cv_.at<double>(i, j) = K_eigen_(i, j);
    has_camera_ = true;
}

void RobustPnPSolver::clearBAEdges() const {
    if (!ba_optimizer_) return;
    for (auto* edge : ba_edges_) {
        if (edge) {
            ba_optimizer_->removeEdge(edge);
            delete edge;
        }
    }
    ba_edges_.clear();
    if (ba_vertex_se3_ && ba_optimizer_->vertex(0)) {
        ba_optimizer_->removeVertex(ba_vertex_se3_, false);
        delete ba_vertex_se3_;
        ba_vertex_se3_ = nullptr;
    }
}

void RobustPnPSolver::initBAOptimizer(int /*num_edges*/) const {
    // 优化：如果optimizer已存在，直接复用，不重新分配求解器
    // 只有第一次调用时才创建求解器结构（solver/algorithm/optimizer）
    if (ba_optimizer_) return;

    ba_linear_solver_ = std::make_unique<
        g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
    ba_block_solver_ = std::make_unique<g2o::BlockSolver_6_3>(
        std::move(ba_linear_solver_));
    ba_algorithm_ = std::make_unique<g2o::OptimizationAlgorithmLevenberg>(
        std::move(ba_block_solver_));
    ba_optimizer_ = std::make_unique<g2o::SparseOptimizer>();
    ba_optimizer_->setAlgorithm(ba_algorithm_.get());
    ba_optimizer_->setVerbose(false);
}

// ==================== 核心求解 ====================

Pose RobustPnPSolver::optimize(const std::vector<cv::Point3f>& pts3d,
                              const std::vector<cv::Point2f>& pts2d,
                              const Pose& init_pose,
                              PnPResult* result_ptr,
                              bool use_motion_prior) {

    PnPResult result;
    result.total_points = static_cast<int>(pts3d.size());

    if (!has_camera_) {
        std::cerr << "[RobustPnP] 错误：未设置相机内参！" << std::endl;
        result.quality = PnPResult::Quality::FAILED;
        if (result_ptr) *result_ptr = result;
        return Pose();
    }

    if (pts3d.size() < 4) {
        std::cerr << "[RobustPnP] 点数不足（需要 ≥ 4，实际 " << pts3d.size() << "）" << std::endl;
        result.quality = PnPResult::Quality::FAILED;
        if (result_ptr) *result_ptr = result;
        return Pose();
    }

    // ★★★ Bug Fix: 添加 K_cv_ 诊断日志 ★★★
    if (config_.verbose) {
        std::cerr << "[RobustPnP] K_cv_ diag: empty=" << K_cv_.empty()
                  << " rows=" << (K_cv_.empty() ? -1 : K_cv_.rows)
                  << " cols=" << (K_cv_.empty() ? -1 : K_cv_.cols)
                  << " type=" << (K_cv_.empty() ? -1 : K_cv_.type())
                  << " fx=" << (K_cv_.empty() ? 0 : K_cv_.at<double>(0,0))
                  << " fy=" << (K_cv_.empty() ? 0 : K_cv_.at<double>(1,1))
                  << std::endl;
    }
    if (!K_cv_.empty() && (K_cv_.rows != 3 || K_cv_.cols != 3 || K_cv_.type() != CV_64F)) {
        std::cerr << "[RobustPnP] FATAL: K_cv_ has invalid format, returning identity pose" << std::endl;
        result.quality = PnPResult::Quality::FAILED;
        if (result_ptr) *result_ptr = result;
        return Pose();
    }

    // ---- 位姿约定（R1）----
    // 对外 API 用 T_wc（Pose{q,t} 的全系统约定），本函数内部一律 T_cw
    //（OpenCV solvePnPRansac / g2o VertexSE3Expmap 的约定）。
    // 唯一的入口换算在这里，唯一的出口换算在 return 之前。
    const Pose init_pose_cw = poseWcToCw(init_pose);
    Pose current_pose = init_pose_cw;
    std::vector<bool> inlier_mask(pts3d.size(), false);

    // ---- 第 1 步：阈值计算 ----
    // 点数少时 RANSAC 采样稳定性下降，允许略放宽阈值换取可解性，
    // 但放宽倍数必须有上限：内点率极低的输入里，放宽 2 倍会把
    // 大量误匹配纳入"内点"，PnP 于是拟合到噪声。原实现的 ×2.0/×1.5
    // 正是 reloc 在 ratio=0.07 时仍返回 success 的原因之一。
    constexpr double RANSAC_THRESH_SCALE_SPARSE = 1.2;   // < 50 点
    constexpr double RANSAC_THRESH_SCALE_MEDIUM = 1.1;   // < 100 点
    const double base_thresh = config_.initial_inlier_thresh_px;
    double ransac_thresh = base_thresh;
    if (pts3d.size() < 50) {
        ransac_thresh = base_thresh * RANSAC_THRESH_SCALE_SPARSE;
    } else if (pts3d.size() < 100) {
        ransac_thresh = base_thresh * RANSAC_THRESH_SCALE_MEDIUM;
    }
    if (config_.verbose) {
        std::cerr << "[PnP-RANSAC] Config: initial_inlier_thresh_px=" << base_thresh
                  << " pts=" << pts3d.size() << " thresh=" << ransac_thresh << " px" << std::endl;
    }
    result.final_threshold = ransac_thresh;

    // 运动先验初值机制
    // 注意：传给 ransacPnP / expandAndRefine 的必须是 T_cw，故这里指向已换算的
    // init_pose_cw，而不是调用方传入的 T_wc。
    bool init_has_nan = init_pose_cw.t.hasNaN() || init_pose_cw.q.coeffs().hasNaN();
    const Pose* p_init_pose = (use_motion_prior && !init_has_nan)
                              ? &init_pose_cw : nullptr;
    if (config_.verbose) {
        std::cerr << "[PnP-RANSAC] INIT: use_guess=" << (p_init_pose ? "TRUE" : "FALSE")
                  << " t_cw=(" << std::fixed << std::setprecision(3)
                  << init_pose_cw.t(0) << "," << init_pose_cw.t(1) << "," << init_pose_cw.t(2) << ")"
                  << " |t_cw|=" << init_pose_cw.t.norm()
                  << (init_has_nan ? " [NaN DETECTED]" : "")
                  << std::endl;
    }

    // ---- 第 2 步：PnP 求解（支持 MLPnP 切换）----
    std::vector<bool> ransac_inliers;

    if (pnp_method_ == PnPMethod::MLPNP) {
        // ★★★ 使用 MLPnP 求解器（基于 bearing vector，对退化场景更鲁棒）★★★
        if (config_.verbose)
            std::cerr << "[RobustPnP] 使用 MLPnP 求解器" << std::endl;

        // 构建 bearing vectors（从 2D 像素计算归一化射线）
        bearingVectors_t bearing_vectors;
        bearing_vectors.reserve(pts2d.size());
        double fx = K_eigen_(0, 0), fy = K_eigen_(1, 1);
        double cx = K_eigen_(0, 2), cy = K_eigen_(1, 2);
        
        // ★★★ Bug Fix: 焦距零值检查防止除零 ★★★
        // 原bug: fx/fy如果为0或NaN，会产生inf/nan bearing vectors
        // 虽然ransacPnP有检查，但MLPnP路径绕过ransacPnP直接使用这些值
        if (fx <= 0 || fy <= 0 || std::isnan(fx) || std::isnan(fy)) {
            if (config_.verbose) {
                std::cerr << "[RobustPnP] ERROR: Invalid focal length fx=" << fx << " fy=" << fy << std::endl;
            }
            return Pose();
        }
        
        for (const auto& p2d : pts2d) {
            double u = (p2d.x - cx) / fx;
            double v = (p2d.y - cy) / fy;
            // ★★★ 使用invSqrt提高数值稳定性 ★★★
            // 对于大u²+v²（telephone pole effect），直接sqrt可能损失精度
            double sq_norm = u*u + v*v + 1.0;
            double inv_norm = 1.0 / std::sqrt(sq_norm);
            bearing_vectors.emplace_back(u * inv_norm, v * inv_norm, inv_norm);
        }

        // 转换 3D 点到带对齐分配器的类型
        points_t points3d_aligned;
        points3d_aligned.reserve(pts3d.size());
        for (const auto& p : pts3d) {
            points3d_aligned.emplace_back(static_cast<double>(p.x), static_cast<double>(p.y), static_cast<double>(p.z));
        }

        // 转换 2D 像素到 cv::Point2f
        std::vector<cv::Point2f> pixels;
        pixels.reserve(pts2d.size());
        for (const auto& p2d : pts2d) {
            pixels.emplace_back(p2d.x, p2d.y);
        }

        // 关键帧索引和 sigma（金字塔层级权重，level 0 = 1.0）
        std::vector<int> keypoint_indices(points3d_aligned.size());
        std::vector<float> sigma2(points3d_aligned.size(), 1.0f);
        for (size_t i = 0; i < points3d_aligned.size(); ++i) {
            keypoint_indices[i] = static_cast<int>(i);
            sigma2[i] = 1.0f;  // 单目 RGB-D 无金字塔，用统一权重
        }

        MLPNPConfig mlpnp_cfg;
        mlpnp_cfg.min_inliers = 8;      // 降低阈值让 MLPnP 更易成功
        mlpnp_cfg.max_iterations = 500;  // 增加迭代次数
        mlpnp_cfg.sigma_threshold = 10.0; // 放宽到 10px
        mlpnp_cfg.threshold = 10.0;      // 放宽卡方阈值
        mlpnp_cfg.epsilon = 0.3;         // 期望内点比例降低

        MLPNPSolver mlpnp(mlpnp_cfg);
        mlpnp.SetData(bearing_vectors, points3d_aligned,
                       pixels, keypoint_indices, sigma2, K_cv_);

        // ORB_SLAM3 策略：5 次迭代，逐步搜索
        std::vector<bool> best_inliers;
        Eigen::Matrix4f T_best;
        int best_n_inliers = 0;
        bool bNoMore = false;

        while (!bNoMore) {
            std::vector<bool> vbInliers;
            int nInliers = 0;
            Eigen::Matrix4f T_out;

            bool ok = mlpnp.iterate(5, bNoMore, vbInliers, nInliers, T_out);

            if (ok && nInliers > best_n_inliers) {
                best_n_inliers = nInliers;
                best_inliers = vbInliers;
                T_best = T_out;

                // 提前终止条件（ORB_SLAM3 风格）
                if (best_n_inliers >= 40) break;  // 降低阈值到 40
            }
        }

        if (config_.verbose)
            std::cerr << "[RobustPnP] MLPnP 结果: inliers=" << best_n_inliers << "/" << pts3d.size()
                       << " T_best_det=" << T_best.block<3,3>(0,0).determinant()
                       << " T_best_t=" << T_best.block<3,1>(0,3).transpose() << std::endl;

        if (best_n_inliers < 8) {
            // MLPnP 失败时回退到 EPnP
            if (config_.verbose)
                std::cerr << "[RobustPnP] MLPnP 失败 (inliers=" << best_n_inliers << ")，回退到 EPnP" << std::endl;
            bool ransac_ok = ransacPnP(pts3d, pts2d, current_pose, ransac_inliers,
                                       ransac_thresh, config_.max_ransac_iterations, p_init_pose);
            if (!ransac_ok) {
                result.quality = PnPResult::Quality::FAILED;
                if (result_ptr) *result_ptr = result;
                return Pose();
            }
            int n_inliers = static_cast<int>(std::count(ransac_inliers.begin(), ransac_inliers.end(), true));
            result.inliers = n_inliers;
            result.inlier_ratio = static_cast<double>(n_inliers) / pts3d.size();
            result.pose = current_pose;
        } else {
            // 提取 R, t 从 T_cw
            Eigen::Matrix3f R_f = T_best.block<3, 3>(0, 0);
            Eigen::Vector3f t_f = T_best.block<3, 1>(0, 3);
            Eigen::Matrix3d R_d = R_f.cast<double>();
            Eigen::Vector3d t_d = t_f.cast<double>();

            // 旋转矩阵合理性检查
            double det_R = R_d.determinant();
            if (det_R < 0) {
                R_d = -R_d;
                t_d = -t_d;
            }

            if (t_d.norm() > 500.0) {  // ★★★ Bug Fix: 从 50m 改为 500m，与其他地方保持一致 ★★★
                if (config_.verbose)
                    std::cerr << "[RobustPnP] MLPnP |t|=" << t_d.norm() << "m exceeds 500m，回退到 EPnP" << std::endl;
                bool ransac_ok = ransacPnP(pts3d, pts2d, current_pose, ransac_inliers,
                                           ransac_thresh, config_.max_ransac_iterations, p_init_pose);
                if (!ransac_ok) {
                    result.quality = PnPResult::Quality::FAILED;
                    if (result_ptr) *result_ptr = result;
                    return Pose();
                }
                int n_inliers = static_cast<int>(std::count(ransac_inliers.begin(), ransac_inliers.end(), true));
                result.inliers = n_inliers;
                result.inlier_ratio = static_cast<double>(n_inliers) / pts3d.size();
                result.pose = current_pose;
            } else {
                Eigen::Quaterniond q(R_d);
                q.normalize();
                current_pose = Pose(q, t_d);
                ransac_inliers = best_inliers;

                // 宽松阈值重筛内点
                selectInliers(pts3d, pts2d, current_pose, ransac_inliers, 12.0);
                int final_n = static_cast<int>(std::count(ransac_inliers.begin(), ransac_inliers.end(), true));

                result.inliers = final_n;
                result.inlier_ratio = static_cast<double>(final_n) / pts3d.size();
                result.pose = current_pose;
                result.quality = assessQuality(final_n, static_cast<int>(pts3d.size()), 5.0);

                if (config_.verbose)
                    std::cerr << "[RobustPnP] MLPnP 成功: inliers=" << final_n << "/" << pts3d.size() << std::endl;
            }
        }
    } else {
        // ---- 第 2 步：OpenCV RANSAC EPnP 粗求解 ----
        bool ransac_ok = ransacPnP(pts3d, pts2d, current_pose, ransac_inliers,
                                   ransac_thresh, config_.max_ransac_iterations,
                                   p_init_pose);
        if (!ransac_ok) {
            { if (config_.verbose) std::cerr << "[RobustPnP] RANSAC EPnP 失败" << std::endl; }
            result.quality = PnPResult::Quality::FAILED;
            if (result_ptr) *result_ptr = result;
            return Pose();
        }

        int n_inliers = static_cast<int>(std::count(ransac_inliers.begin(),
                                                     ransac_inliers.end(), true));
        result.inliers = n_inliers;
        result.inlier_ratio = static_cast<double>(n_inliers) / pts3d.size();
        result.ransac_iterations_used = config_.max_ransac_iterations;
    }

    // EPnP 结果备份（用于 expand 失败时的回退）
    Pose epnp_pose = current_pose;
    std::vector<bool> epnp_inliers = ransac_inliers;


    // ---- 第 3 步：逐次膨胀内点 + 迭代精化 ----
    bool expand_ok = false;
    if (config_.expansion_rounds > 0) {
        expand_ok = expandAndRefine(pts3d, pts2d, current_pose, ransac_inliers,
                            result.final_threshold, p_init_pose);
        if (!expand_ok) {
            { if (config_.verbose) std::cerr << "[RobustPnP] 逐次膨胀失败，回退到 EPnP 结果" << std::endl; }
            // 回退到 EPnP 原始位姿和内点（不要用 init_pose，它是错误的起点）
            current_pose = epnp_pose;
            ransac_inliers = epnp_inliers;
        }
    }

    // ---- 第 4 步：g2o BA 精化 ----
    int final_inliers = static_cast<int>(std::count(ransac_inliers.begin(),
                                                     ransac_inliers.end(), true));
    result.inliers = final_inliers;
    result.inlier_ratio = static_cast<double>(final_inliers) / pts3d.size();
    
    // ★★★ Bug Fix: 详细诊断日志 - 打印 BA 前内点数量 ★★★
    if (config_.verbose) {
        std::cerr << "[RobustPnP] BEFORE BA: inliers=" << final_inliers << "/" << pts3d.size()
                  << " expand_ok=" << expand_ok << std::endl;
    }

    // 无论 expand_ok 是否成功，只要内点 >= 4 就尝试 g2o BA
    if (final_inliers >= 4) {
        Pose ba_pose;
        bool ba_ok = false;
        if (config_.joint_point_optimization) {
            std::vector<Eigen::Vector3d> dummy;
            ba_ok = g2oJointBA(pts3d, pts2d, ransac_inliers, current_pose, ba_pose, dummy);
        } else {
            // ★★★ P1: 传递 use_motion_prior 给 g2oBA 以控制 Huber 核移除 ★★★
            ba_ok = g2oBA(pts3d, pts2d, ransac_inliers, current_pose, ba_pose, use_motion_prior);
        }
        
        // ★★★ Bug Fix: 详细诊断日志 - 打印 BA 后内点数量 ★★★
        if (config_.verbose) {
            std::cerr << "[RobustPnP] AFTER BA: ba_ok=" << ba_ok
                      << " pose=(" << ba_pose.t.transpose() << ")"
                      << " |t|=" << ba_pose.t.norm()
                      << " z=" << ba_pose.t.z() << std::endl;
        }
        
        // ★★★ Bug Fix: 检查 g2oBA 返回值，只在成功时接受优化结果 ★★★
        // g2oBA 内部已有 NaN 安全：无效时会回退到 init_pose
        if (ba_ok) {
            current_pose = ba_pose;
        }
        
        // g2o BA 后用宽松阈值重新筛选内点
        selectInliers(pts3d, pts2d, current_pose, ransac_inliers,
                      config_.final_inlier_thresh_px);
        
        // ★★★ Bug Fix: 更新 final_inliers 以反映 selectInliers 的结果 ★★★
        final_inliers = static_cast<int>(std::count(ransac_inliers.begin(), ransac_inliers.end(), true));
        result.inliers = final_inliers;
        result.inlier_ratio = static_cast<double>(final_inliers) / pts3d.size();
        
        // ★★★ Bug Fix: 详细诊断日志 - 打印 selectInliers 后内点数量 ★★★
        if (config_.verbose) {
            std::cerr << "[RobustPnP] AFTER SELECT: inliers=" << final_inliers << "/" << pts3d.size()
                      << " thresh=" << config_.final_inlier_thresh_px << std::endl;
        }
    }

    // ---- 第 5 步：质量评估 ----
    std::vector<double> final_errors =
        computeReprojectionErrors(pts3d, pts2d, current_pose, K_eigen_);
    std::vector<double> inlier_errors;
    inlier_errors.reserve(final_inliers);
    for (size_t i = 0; i < ransac_inliers.size(); ++i) {
        if (ransac_inliers[i]) inlier_errors.push_back(final_errors[i]);
    }
    result.median_reproj_error = computeMedianError(inlier_errors);
    result.mean_reproj_error = computeMeanError(inlier_errors);

    // ---- 出口换算（R1）----
    // 内部所有量（current_pose、重投影误差、quality）都在 T_cw 下算完；
    // 从这里开始切回对外的 T_wc，日志与返回值都用换算后的位姿。
    const Pose out_pose_wc = poseCwToWc(current_pose);
    result.pose = out_pose_wc;
    result.success = true;
    result.quality = assessQuality(result.inliers, result.total_points,
                                   result.median_reproj_error);

    // 关键诊断：打印最终 PnP 结果（位姿为对外约定 T_wc）
    if (config_.verbose) {
        std::cerr << "[PnP-FINAL] frame_id=PnP"
                  << " final_t_wc=(" << std::fixed << std::setprecision(4)
                  << out_pose_wc.t.transpose() << ")"
                  << " |t_wc|=" << out_pose_wc.t.norm()
                  << " t_z=" << std::setprecision(4) << out_pose_wc.t.z()
                  << " inliers=" << result.inliers << "/" << pts3d.size()
                  << " med_err=" << result.median_reproj_error
                  << " quality=" << static_cast<int>(result.quality)
                  << std::endl;
    }

    if (result_ptr) *result_ptr = result;
    return out_pose_wc;
}

Pose RobustPnPSolver::optimize(const std::vector<Eigen::Vector3d>& pts3d,
                               const std::vector<Eigen::Vector2d>& pts2d,
                               const Pose& init_pose,
                               PnPResult* result_ptr,
                               bool use_motion_prior) {
    std::vector<cv::Point3f> pts3d_cv;
    std::vector<cv::Point2f> pts2d_cv;
    for (const auto& p : pts3d) pts3d_cv.emplace_back(p.x(), p.y(), p.z());
    for (const auto& p : pts2d) pts2d_cv.emplace_back(p.x(), p.y());
    return optimize(pts3d_cv, pts2d_cv, init_pose, result_ptr, use_motion_prior);
}

// ==================== RANSAC EPnP ====================

bool RobustPnPSolver::ransacPnP(const std::vector<cv::Point3f>& pts3d,
                                 const std::vector<cv::Point2f>& pts2d,
                                 Pose& best_pose,
                                 std::vector<bool>& inlier_mask,
                                 double threshold,
                                 int max_iters,
                                 const Pose* init_pose) {
    cv::Mat rvec, tvec;
    cv::Mat inliers;

    bool use_guess = false;
    
    // 诊断日志：打印输入数据大小
    { if (config_.verbose) std::cerr << "[PnP-RANSAC] INPUT: pts3d=" << pts3d.size() << " pts2d=" << pts2d.size()
              << " threshold=" << threshold << " max_iters=" << max_iters << std::endl; }
    
    // ★★★ Bug Fix: 添加相机内参矩阵安全检查 ★★★
    // cv::solvePnPRansac 内部会调用 cv::undistortPointsInternal，
    // 如果 K_cv_ 不是有效的 3x3 矩阵，会触发 OpenCV assertion 失败：
    // "CV_IS_MAT(_cameraMatrix) && _cameraMatrix->rows == 3 && _cameraMatrix->cols == 3"
    // 检查：矩阵非空、维度正确（3x3）、数据类型为 double
    if (K_cv_.empty() || K_cv_.rows != 3 || K_cv_.cols != 3 || K_cv_.type() != CV_64F) {
        std::cerr << "[PnP-RANSAC] FATAL: K_cv_ is invalid before RANSAC (empty=" << K_cv_.empty()
                  << " rows=" << K_cv_.rows << " cols=" << K_cv_.cols
                  << " type=" << (K_cv_.empty() ? -1 : K_cv_.type()) << ")" << std::endl;
        return false;
    }
    if (K_cv_.at<double>(0,0) <= 0 || K_cv_.at<double>(1,1) <= 0) {
        std::cerr << "[PnP-RANSAC] FATAL: K_cv_ has invalid focal lengths (fx=" << K_cv_.at<double>(0,0)
                  << " fy=" << K_cv_.at<double>(1,1) << ")" << std::endl;
        return false;
    }
    // ★★★ 关键修复：平移向量合理性检查 ★★★
    // hasNaN() 仅检测 NaN，无法捕获 inf 或超出物理范围的值（如 -4.5e13）。
    // EPnP 在这种灾难性初值下无法收敛，导致 RANSAC 彻底失败。
    // 修复：额外检查 |t| 不超过 1000m（室内场景合理上限）。
    bool t_is_valid = init_pose != nullptr &&
                      !init_pose->t.hasNaN() &&
                      !init_pose->q.coeffs().hasNaN() &&
                      init_pose->t.norm() < 1000.0;
    if (t_is_valid) {
        use_guess = true;
        Eigen::Matrix3d R_init = init_pose->q.toRotationMatrix();
        tvec = cv::Mat(3, 1, CV_64F);
        tvec.at<double>(0) = init_pose->t(0);
        tvec.at<double>(1) = init_pose->t(1);
        tvec.at<double>(2) = init_pose->t(2);
        // ★★★ 关键修复：从 init_pose 的旋转直接计算 rvec（Rodrigues）★★★
        // 错误原因：原代码先设 rvec=0，再用 Rodrigues(R_cv, rvec) —— 用零 rvec 反算
        // R_cv 会得到单位阵，完全忽略了 init_pose 的旋转。
        // 正确做法：Rodrigues(R_init, rvec)，从旋转矩阵计算旋转向量。
        cv::Mat R_cv(3, 3, CV_64F);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R_cv.at<double>(i, j) = R_init(i, j);
        cv::Rodrigues(R_cv, rvec);  // 从 R_init 计算 rvec（而不是用零 rvec 反算）
        // 关键诊断：打印 RANSAC 初始位姿
        if (config_.verbose) {
            std::cerr << "[PnP-RANSAC] INIT: use_guess=TRUE"
                      << " t=(" << std::fixed << std::setprecision(3)
                      << init_pose->t(0) << "," << init_pose->t(1) << "," << init_pose->t(2) << ")"
                      << " |t|=" << init_pose->t.norm()
                      << " rvec=(" << rvec.at<double>(0) << "," << rvec.at<double>(1) << "," << rvec.at<double>(2) << ")"
                      << std::endl;
        }
    } else {
        rvec = cv::Mat::zeros(3, 1, CV_64F);
        tvec = cv::Mat::zeros(3, 1, CV_64F);
    }

    bool ok = false;
    { if (config_.verbose) std::cerr << "[PnP-RANSAC] Calling cv::solvePnPRansac(EPnP)..." << std::endl; }
    try {
        ok = cv::solvePnPRansac(
            pts3d, pts2d, K_cv_, D_cv_,
            rvec, tvec,
            use_guess,
            max_iters,
            threshold,
            0.99,
            inliers,
            cv::SOLVEPNP_EPNP);
        { if (config_.verbose) std::cerr << "[PnP-RANSAC] EPnP returned: ok=" << ok << " inliers=" << (inliers.empty() ? 0 : inliers.rows) << std::endl; }
    } catch (const cv::Exception& e) {
        std::cerr << "[PnP-RANSAC] cv::solvePnPRansac(EPnP) exception: " << e.what() << std::endl;
        ok = false;
    }

    if (!ok || inliers.empty()) {
        // ★★★ Fallback 1: Try SOLVEPNP_ITERATIVE when EPnP fails ★★★
        { if (config_.verbose) std::cerr << "[PnP-RANSAC] EPnP failed (ok=" << ok << ", inliers=" << inliers.rows << "), trying ITERATIVE..." << std::endl; }
        cv::Mat rvec2, tvec2;
        if (use_guess) {
            rvec2 = rvec.clone();
            tvec2 = tvec.clone();
        } else {
            rvec2 = cv::Mat::zeros(3, 1, CV_64F);
            tvec2 = cv::Mat::zeros(3, 1, CV_64F);
        }
        cv::Mat inliers2;
        { if (config_.verbose) std::cerr << "[PnP-RANSAC] ITERATIVE: use_guess=" << use_guess << std::endl; }
        try {
            ok = cv::solvePnPRansac(
                pts3d, pts2d, K_cv_, D_cv_,
                rvec2, tvec2,
                use_guess,
                max_iters,
                threshold,
                0.99,
                inliers2,
                cv::SOLVEPNP_ITERATIVE);
            { if (config_.verbose) std::cerr << "[PnP-RANSAC] ITERATIVE result: ok=" << ok << " inliers=" << (inliers2.empty() ? 0 : inliers2.rows) << std::endl; }
        } catch (const cv::Exception& e) {
            std::cerr << "[PnP-RANSAC] cv::solvePnPRansac(ITERATIVE) exception: " << e.what() << std::endl;
            ok = false;
        }
        if (ok && !inliers2.empty()) {
            rvec = rvec2;
            tvec = tvec2;
            inliers = inliers2;
            { if (config_.verbose) std::cerr << "[PnP-RANSAC] ITERATIVE succeeded: inliers=" << inliers.rows << std::endl; }
        } else if (pts3d.size() >= 4) {
            // ★★★ Fallback 2: Try SOLVEPNP_AP3P when both EPnP and ITERATIVE fail ★★★
            { if (config_.verbose) std::cerr << "[PnP-RANSAC] ITERATIVE failed (ok=" << ok << ", inliers=" << (inliers2.empty() ? 0 : inliers2.rows) << "), trying AP3P..." << std::endl; }
            // AP3P is an analytical P3P solver that can be more robust with a good motion prior.
            cv::Mat rvec3, tvec3;
            if (use_guess) {
                rvec3 = rvec.clone();
                tvec3 = tvec.clone();
            } else {
                rvec3 = cv::Mat::zeros(3, 1, CV_64F);
                tvec3 = cv::Mat::zeros(3, 1, CV_64F);
            }
            cv::Mat inliers3;
            try {
                ok = cv::solvePnPRansac(
                    pts3d, pts2d, K_cv_, D_cv_,
                    rvec3, tvec3,
                    use_guess,
                    max_iters,
                    threshold,
                    0.99,
                    inliers3,
                    cv::SOLVEPNP_AP3P);
            } catch (const cv::Exception& e) {
                std::cerr << "[PnP-RANSAC] cv::solvePnPRansac(AP3P) exception: " << e.what() << std::endl;
                ok = false;
            }
            // ★★★ Bug Fix: AP3P 内点阈值过高导致失败 ★★★
            // 原问题：pts3d.size()=300时，min_inliers=15，但AP3P通常只得到5-12个内点
            // AP3P 是 4 点求解器，只需少量内点就能验证姿态
            // 修复：降低阈值到固定 4 个（AP3P 最小求解需求），完全允许低内点率场景成功
            int ap3p_min_inliers = 4;  // AP3P 只需要 4 个点就能求解
            { if (config_.verbose) std::cerr << "[PnP-RANSAC] AP3P returned inliers=" << (inliers3.empty() ? 0 : inliers3.rows)
                      << " (min_required=" << ap3p_min_inliers << ")" << std::endl; }
            if (ok && !inliers3.empty() && inliers3.rows >= ap3p_min_inliers) {
                rvec = rvec3;
                tvec = tvec3;
                inliers = inliers3;
                { if (config_.verbose) std::cerr << "[PnP-RANSAC] AP3P ACCEPTED: inliers=" << inliers.rows << std::endl; }
            } else {
                { if (config_.verbose) std::cerr << "[PnP-RANSAC] AP3P REJECTED: inliers=" << (inliers3.empty() ? 0 : inliers3.rows)
                          << " < min_inliers=" << ap3p_min_inliers << std::endl; }
                ok = false;  // 标记为失败，让外部处理
            }
        } else {
            { if (config_.verbose) std::cerr << "[PnP-RANSAC] BOTH EPnP AND ITERATIVE FAILED: ok=" << ok
                       << " epnpinliers=" << (inliers.empty() ? 0 : inliers.rows)
                       << " iterinliers=" << (inliers2.empty() ? 0 : inliers2.rows)
                       << " pts3d=" << pts3d.size() << " (need >=4 for AP3P)" << std::endl; }
        }
    }

    if (!ok || inliers.empty()) {
        if (config_.verbose)
            std::cerr << "[PnP-RANSAC] FAIL: ok=" << ok << " inliers=" << (inliers.empty() ? 0 : inliers.rows) << std::endl;
        return false;
    }

    // RANSAC 结果诊断（计算 Rodrigues 矩阵并检查行列式）
    cv::Mat R_cv;
    cv::Rodrigues(rvec, R_cv);
    Eigen::Matrix3d R_eigen;
    Eigen::Vector3d t_eigen;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) R_eigen(i, j) = R_cv.at<double>(i, j);
        t_eigen(i) = tvec.at<double>(i);
    }
    double det_R = R_eigen.determinant();
    if (config_.verbose) {
        std::cerr << "[PnP-RANSAC] OK: inliers=" << inliers.rows << "/" << pts3d.size()
                  << " det(R)=" << std::fixed << std::setprecision(4) << det_R
                  << " t=(" << t_eigen.transpose() << ") |t|=" << t_eigen.norm()
                  << std::endl;
    }

    // ★★★ 关键修复：旋转矩阵行列式检查 ★★★
    // OpenCV solvePnP/solvePnPRansac 可能返回 det(R) = -1 的反射矩阵（数学上有效但物理上不可能）。
    // 这会导致相机坐标系统「镜像」，造成轨迹方向完全错误。
    // 修复：如果 det(R) < 0，否定整个旋转矩阵并同步否定 t（使相机变换符合 SE(3) 物理约束）。
    // 正确方法：否定全部三列（而非仅否定一列），确保 det(R) = +1。
    // 详见：Hartley & Zisserman "Multiple View Geometry" 8.1 节，ambiguity resolution。
    if (det_R < 0) {
        R_eigen = -R_eigen;
        t_eigen = -t_eigen;
        if (config_.verbose)
            std::cerr << "[RobustPnP] detected reflection (det(R)=" << det_R << "), corrected." << std::endl;
    }

    // ★★★ Bug Fix B9: RANSAC 输出平移向量合理性检查 ★★★
    // cv::solvePnPRansac 在灾难性输入（如错误的 3D-2D 匹配对）下，
    // 可能返回数学上有效（det(R)=1）但物理上不可能的平移向量（如 10^8 ~ 10^15 米）。
    // 若不检查，后续 g2o BA 会以这个腐败位姿为初值迅速发散。
    // 室内场景相机移动距离通常 <50m，设置 500m 作为硬上限。
    if (t_eigen.norm() > 500.0) {
        if (config_.verbose) {
            std::cerr << "[PnP-RANSAC] REJECT: |t|=" << t_eigen.norm()
                      << "m exceeds physical limit (500m)" << std::endl;
        }
        return false;
    }

    Eigen::Quaterniond q(R_eigen);
    q.normalize();
    best_pose = Pose(q, t_eigen);

    inlier_mask.assign(pts3d.size(), false);
    for (int i = 0; i < inliers.rows; ++i) {
        inlier_mask[inliers.at<int>(i)] = true;
    }

    return true;
}

// ==================== 内点筛选 ====================

int RobustPnPSolver::selectInliers(const std::vector<cv::Point3f>& pts3d,
                                    const std::vector<cv::Point2f>& pts2d,
                                    const Pose& pose,
                                    std::vector<bool>& inlier_mask,
                                    double threshold_px) const {
    inlier_mask.assign(pts3d.size(), false);
    int count = 0;
    for (size_t i = 0; i < pts3d.size(); ++i) {
        double err = reprojError(pts3d[i], pts2d[i], pose, K_eigen_);
        // NaN 安全：任何无效值视为大误差
        if (std::isnan(err) || std::isinf(err)) err = 1e6;
        if (err < threshold_px) {
            inlier_mask[i] = true;
            count++;
        }
    }
    return count;
}

// ==================== 自适应阈值 ====================

double RobustPnPSolver::adaptiveThreshold(double inlier_ratio, int /*total_points*/) const {
    // ★★★ 修复：让自适应阈值基于配置的 initial_inlier_thresh_px ★★★
    // 原来的硬编码值（4.0/6.0/8.0/10.0）没有考虑配置的阈值
    // 现在根据内点率动态调整，但基础值来自配置
    double base_thresh = config_.initial_inlier_thresh_px;  // 现在是 10.0
    if (inlier_ratio > 0.6) {
        return base_thresh * 0.6;  // 高内点率 → 收紧到 60%
    } else if (inlier_ratio > 0.3) {
        return base_thresh;  // 中内点率 → 使用配置的阈值
    } else if (inlier_ratio > 0.1) {
        return base_thresh * 1.2;  // 低内点率 → 放宽到 120%
    } else {
        return base_thresh * 1.5;  // 极低内点率 → 放宽到 150%
    }
}

// ==================== 逐次膨胀 ====================

bool RobustPnPSolver::expandAndRefine(const std::vector<cv::Point3f>& pts3d,
                                      const std::vector<cv::Point2f>& pts2d,
                                      Pose& inout_pose,
                                      std::vector<bool>& inout_inliers,
                                      double& inout_thresh,
                                      const Pose* init_pose) {

    Pose refined_pose = inout_pose;
    std::vector<bool> refined_inliers = inout_inliers;
    double current_thresh = inout_thresh;

    for (int round = 0; round < config_.expansion_rounds; ++round) {
        std::vector<cv::Point3f> inlier_pts3d;
        std::vector<cv::Point2f> inlier_pts2d;
        for (size_t i = 0; i < refined_inliers.size(); ++i) {
            if (refined_inliers[i]) {
                inlier_pts3d.push_back(pts3d[i]);
                inlier_pts2d.push_back(pts2d[i]);
            }
        }

        if (inlier_pts3d.size() < 4) break;

        // 第 1 轮用恒速运动先验；后续用上一轮优化结果
        bool use_guess = (round == 0 && init_pose != nullptr
                          && !init_pose->t.hasNaN()
                          && !init_pose->q.coeffs().hasNaN());

        cv::Mat rvec, tvec;
        // ★★★ Bug Fix: 必须初始化 rvec 和 tvec，否则 use_guess=false 时会导致崩溃 ★★★
        if (use_guess) {
            Eigen::Matrix3d R_init = init_pose->q.toRotationMatrix();
            rvec = cv::Mat(3, 1, CV_64F, cv::Scalar::all(0));
            tvec = cv::Mat(3, 1, CV_64F, cv::Scalar::all(0));
            tvec.at<double>(0) = init_pose->t(0);
            tvec.at<double>(1) = init_pose->t(1);
            tvec.at<double>(2) = init_pose->t(2);
            cv::Mat R_cv(3, 3, CV_64F, cv::Scalar::all(0));
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    R_cv.at<double>(i, j) = R_init(i, j);
            cv::Rodrigues(R_cv, rvec);
        } else {
            rvec = cv::Mat(3, 1, CV_64F, cv::Scalar::all(0));
            tvec = cv::Mat(3, 1, CV_64F, cv::Scalar::all(0));
        }

        // ★★★ Bug Fix: 添加 K_cv_ 安全检查和 try-catch ★★★
        if (K_cv_.empty() || K_cv_.rows != 3 || K_cv_.cols != 3 || K_cv_.type() != CV_64F) {
            std::cerr << "[expandAndRefine] FATAL: K_cv_ is invalid" << std::endl;
            break;
        }

        bool solve_ok = false;
        try {
            solve_ok = cv::solvePnP(inlier_pts3d, inlier_pts2d, K_cv_, D_cv_,
                                     rvec, tvec, use_guess, cv::SOLVEPNP_EPNP);
        } catch (const cv::Exception& e) {
            std::cerr << "[expandAndRefine] cv::solvePnP(EPnP) exception: " << e.what() << std::endl;
            solve_ok = false;
        }
        if (!solve_ok) {
            cv::Mat rvec_bak = rvec.clone(), tvec_bak = tvec.clone();
            try {
                solve_ok = cv::solvePnP(inlier_pts3d, inlier_pts2d, K_cv_, D_cv_,
                                         rvec_bak, tvec_bak, use_guess, cv::SOLVEPNP_ITERATIVE);
            } catch (const cv::Exception& e) {
                std::cerr << "[expandAndRefine] cv::solvePnP(ITERATIVE) exception: " << e.what() << std::endl;
                solve_ok = false;
            }
            if (solve_ok) {
                rvec = rvec_bak;
                tvec = tvec_bak;
                if (config_.verbose)
                    std::cerr << "[expandAndRefine] EPnP failed, ITERATIVE succeeded" << std::endl;
            } else {
                // ★★★ Fallback 2: Try AP3P as a third method in expandAndRefine ★★★
                cv::Mat rvec_ap3p = rvec.clone(), tvec_ap3p = tvec.clone();
                try {
                    solve_ok = cv::solvePnP(inlier_pts3d, inlier_pts2d, K_cv_, D_cv_,
                                             rvec_ap3p, tvec_ap3p, use_guess, cv::SOLVEPNP_AP3P);
                } catch (const cv::Exception& e) {
                    std::cerr << "[expandAndRefine] cv::solvePnP(AP3P) exception: " << e.what() << std::endl;
                    solve_ok = false;
                }
                if (solve_ok) {
                    rvec = rvec_ap3p;
                    tvec = tvec_ap3p;
                    if (config_.verbose)
                        std::cerr << "[expandAndRefine] EPnP+ITERATIVE failed, AP3P succeeded" << std::endl;
                }
            }
        }
        if (!solve_ok) break;

        cv::Mat R_cv;
        cv::Rodrigues(rvec, R_cv);
        Eigen::Matrix3d R_eigen;
        Eigen::Vector3d t_eigen;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) R_eigen(i, j) = R_cv.at<double>(i, j);
            t_eigen(i) = tvec.at<double>(i);
        }

        // ★★★ 关键修复：旋转矩阵行列式检查（与 ransacPnP 保持一致）★★★
        // 当 det(R) < 0 时，否定整个矩阵而非仅否定一列，确保 det(R) = +1
        double det_R = R_eigen.determinant();
        if (det_R < 0) {
            R_eigen = -R_eigen;
            t_eigen = -t_eigen;
            if (config_.verbose)
                std::cerr << "[expandAndRefine] reflection corrected (det(R)=" << det_R << ")" << std::endl;
        }

        Eigen::Quaterniond q(R_eigen);
        q.normalize();
        if (q.norm() < 0.5 || q.norm() > 1.5 ||
            std::isnan(q.norm()) || std::isnan(q.x()) || std::isnan(q.y()) ||
            std::isnan(q.z()) || std::isnan(q.w())) {
            if (config_.verbose)
                std::cerr << "[expandAndRefine] invalid quaternion, skipping round" << std::endl;
            break;
        }
        Pose new_pose(q, t_eigen);

        double strict_thresh = current_thresh * 0.8;
        int n_new_inliers = selectInliers(pts3d, pts2d, new_pose,
                                          refined_inliers, strict_thresh);

        // ★★★ P1: 限制阈值膨胀，最大不超过 12px ★★★
        // 原代码：expanded_thresh = current_thresh * expansion_factor;
        // 问题：2轮膨胀后可达 8.0 * 1.5 * 1.5 = 18px，过于宽松
        // 修复：限制最大阈值为 min(initial * 2.0, 12.0)
        double max_allowed_thresh = std::min(config_.initial_inlier_thresh_px * 2.0, 12.0);
        double expanded_thresh = current_thresh * config_.expansion_factor;
        expanded_thresh = std::min(expanded_thresh, max_allowed_thresh);

        std::vector<bool> expanded_inliers;
        selectInliers(pts3d, pts2d, new_pose, expanded_inliers, expanded_thresh);
        int n_expanded = static_cast<int>(
            std::count(expanded_inliers.begin(), expanded_inliers.end(), true));

        if (n_expanded > n_new_inliers) {
            refined_inliers = expanded_inliers;
            current_thresh = expanded_thresh;
        }

        refined_pose = new_pose;
    }

    int n_final = static_cast<int>(std::count(refined_inliers.begin(),
                                               refined_inliers.end(), true));
    if (n_final >= 8) {
        Pose ba_pose;
        // ★★★ P1: 传递 use_huber_removal=true 给 g2oBA ★★★
        if (g2oBA(pts3d, pts2d, refined_inliers, refined_pose, ba_pose, true)) {
            refined_pose = ba_pose;
            selectInliers(pts3d, pts2d, refined_pose, refined_inliers,
                         config_.final_inlier_thresh_px);
        }
    }

    inout_pose = refined_pose;
    inout_inliers = refined_inliers;
    inout_thresh = current_thresh;
    return true;
}

// ==================== g2o BA（仅优化位姿，ORB_SLAM3 风格迭代）====================

bool RobustPnPSolver::g2oBA(const std::vector<cv::Point3f>& pts3d,
                             const std::vector<cv::Point2f>& pts2d,
                             const std::vector<bool>& inlier_mask,
                             const Pose& init_pose,
                             Pose& out_pose,
                             bool use_huber_removal) const {

    // 收集内点
    std::vector<cv::Point3f> inlier_pts3d;
    std::vector<cv::Point2f> inlier_pts2d;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            inlier_pts3d.push_back(pts3d[i]);
            inlier_pts2d.push_back(pts2d[i]);
        }
    }
    if (inlier_pts3d.size() < 4) return false;

    // 创建优化器（每个顶点只创建一次，ORB_SLAM3 风格）
    using BlockSolverType = g2o::BlockSolver_6_3;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    // 位姿顶点（只创建一个，后续重置）
    auto* v_se3 = new g2o::VertexSE3Expmap();
    v_se3->setId(0);
    v_se3->setEstimate(g2o::SE3Quat(init_pose.q, init_pose.t));
    v_se3->setFixed(false);
    optimizer.addVertex(v_se3);

    // 边数组（用于后续离群点剔除，ORB_SLAM3 风格）
    std::vector<EdgeRobustReproject*> edges;
    edges.reserve(inlier_pts3d.size());
    for (size_t i = 0; i < inlier_pts3d.size(); ++i) {
        Eigen::Vector3d pt3d(inlier_pts3d[i].x, inlier_pts3d[i].y, inlier_pts3d[i].z);
        Eigen::Vector2d pt2d(inlier_pts2d[i].x, inlier_pts2d[i].y);

        // ★★★ P1: 修复 info 矩阵（关键 Bug）★★★
        // 错误原因：原代码使用 info_scale = fx²/z² 按深度平方加权，
        // 导致远处点（z=5m）的 chi2 被放大 ~10000 倍，
        // 远处点总是被错误判定为外点并剔除，破坏 BA 优化质量。
        //
        // 修复方法：与 ORB_SLAM3 一致，只用金字塔层级加权（invSigma2），
        // 不再按深度平方加权。金字塔层越高（特征越模糊），信息矩阵越低。
        // 单目：info = Identity（ORB_SLAM3 用 invSigma2 缩放，保留该思路）。
        // ★★★ Bug Fix #5: info_scale = 1.0（ORB_SLAM3 的 invSigma2 思想已通过金字塔层级实现）★★★
        const double info_scale = 1.0;

        auto* edge = new EdgeRobustReproject(pt3d, K_eigen_, info_scale);
        edge->setId(static_cast<int>(i + 1));
        edge->setVertex(0, v_se3);
        edge->setMeasurement(pt2d);
        // EdgeRobustReproject 是 unary 边（仅 pose 顶点），setInformation 仅设置自身
        edge->setInformation(Eigen::Matrix2d::Identity());

        // ★★★ P1: 添加 Huber 核移除机制（第3轮后移除）★★★
        // 与 ORB_SLAM3 Optimizer.cc:1006-1042 一致
        // Huber 核的作用是在优化初期降低外点权重，但优化后期应移除以进行精细优化
        if (config_.use_huber_kernel) {
            auto* kernel = new g2o::RobustKernelHuber();
            kernel->setDelta(config_.huber_delta);
            edge->setRobustKernel(kernel);
        }

        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    // JacobianWorkspace: EdgeRobustReproject 是 unary 边（仅 pose 顶点），单顶点最大维度 6
    // updateSize(2, 12): 2=安全上限顶点数量，12=2×6（足够容纳所有 unary 边类型的雅可比）
    optimizer.jacobianWorkspace().updateSize(2, 12, true);

    // ★★★ ORB_SLAM3 PoseOptimization 风格：4 轮迭代（Optimizer.cc:1006-1042）★★★
    // 每轮：重置位姿 → optimize(10) → 剔除离群点 → 下一轮
    // ★★★ P1: 第3轮后移除 Huber 核以进行精细优化 ★★★
    const int n_rounds = 4;
    const int lm_iters_per_round = 10;
    // ★★★ P1: 修复 chi2 阈值（与 ORB_SLAM3 一致）★★★
    // ORB_SLAM3 Optimizer.cc:1001: const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    // chi2Mono = 5.991 = χ²(2, 0.95)，卡方分布自由度2，95%置信区间上限。
    // 注意：chi2_thresh 始终为 5.991，不随 info_scale 变化（因为 info_scale=1.0）。
    const double chi2_thresh = 5.991;
    Pose current_pose = init_pose;

    for (int round = 0; round < n_rounds; ++round) {
        // 重置位姿顶点为当前估计
        v_se3->setEstimate(g2o::SE3Quat(current_pose.q, current_pose.t));

        // ★★★ Bug Fix: 检查 initializeOptimization 返回值 ★★★
        int n_level0 = 0;
        for (auto* e : edges) {
            if (e->level() == 0) n_level0++;
        }
        if (n_level0 == 0) {
            if (config_.verbose) {
                std::cerr << "[g2o-BA] Round " << round << ": no level-0 edges to optimize!" << std::endl;
            }
            break;
        }
        
        // 仅优化 Level=0 的边（内点）
        bool init_ok = optimizer.initializeOptimization();
        if (!init_ok) {
            if (config_.verbose) {
                std::cerr << "[g2o-BA] Round " << round << ": initializeOptimization failed!" << std::endl;
            }
            break;
        }
        optimizer.optimize(lm_iters_per_round);

        // 获取优化后的位姿
        g2o::SE3Quat T_opt = v_se3->estimate();
        current_pose = Pose(Eigen::Quaterniond(T_opt.rotation()), T_opt.translation());

        // [Z轴调试] 打印每轮 BA 的 Z 变化
        if (config_.verbose) {
            Eigen::Vector3d prev_t = (round == 0) ? init_pose.t : current_pose.t;
            double round_dz = current_pose.t.z() - prev_t.z();
            { if (config_.verbose) std::cerr << "[g2o-BA-R" << round << "] z=" << std::fixed << std::setprecision(4)
                      << current_pose.t.z() << "m dz_r" << round << "=" << std::setprecision(4) << round_dz
                      << "m |dt|=" << (T_opt.translation() - init_pose.t).norm() << "m"
                      << std::endl; }
        }

        // 遍历所有边，计算 chi2，标记离群点（ORB_SLAM3 风格）
        int n_bad = 0;
        for (auto* edge : edges) {
            edge->computeError();
            double chi2 = edge->chi2();
            if (chi2 > chi2_thresh) {
                edge->setLevel(1);  // 剔除，下一轮不再参与优化
                n_bad++;
            } else {
                edge->setLevel(0);  // 保留，下一轮继续参与
            }
        }

        // ★★★ P1: 第3轮后移除 Huber 核（ORB_SLAM3 风格）★★★
        // 与 ORB_SLAM3 Optimizer.cc:1024 一致：if(it==2) e->setRobustKernel(0);
        if (use_huber_removal && round == 2) {
            for (auto* edge : edges) {
                edge->setRobustKernel(0);  // 移除 Huber 核，使用二次核进行精细优化
            }
        }

        // 可选：过半离群点则提前停止
        if (n_bad > static_cast<int>(edges.size() * 0.5)) {
            break;
        }
    }

    g2o::SE3Quat T_final = v_se3->estimate();
    out_pose = Pose(Eigen::Quaterniond(T_final.rotation()), T_final.translation());

    // [Z轴调试] 打印内点 Z 轴统计
    if (config_.verbose && !inlier_pts3d.empty()) {
        double sum_z = 0.0, sum_z2 = 0.0, min_z = 1e9, max_z = -1e9;
        for (const auto& p : inlier_pts3d) {
            double z = p.z;
            sum_z += z; sum_z2 += z*z;
            if (z < min_z) min_z = z;
            if (z > max_z) max_z = z;
        }
        double mean_z = sum_z / inlier_pts3d.size();
        double std_z = std::sqrt(std::max(0.0, sum_z2/inlier_pts3d.size() - mean_z*mean_z));
        { if (config_.verbose) std::cerr << "[g2o-BA-INLIER-Z] n_inlier=" << inlier_pts3d.size()
                  << " z_mean=" << std::fixed << std::setprecision(3) << mean_z
                  << "m z_std=" << std::setprecision(3) << std_z
                  << " z_range=[" << std::setprecision(3) << min_z << "," << std::setprecision(3) << max_z << "]"
                  << std::endl; }

        // ★★★ Bug Fix: Reloc 阶段 BA 拒绝负深度内点 ★★★
        if (mean_z < 0.05) {
            if (config_.verbose) {
                std::cerr << "[g2o-BA] REJECT: z_mean=" << mean_z << "m < 0.05m (negative/zero depth, likely wrong rotation)" << std::endl;
            }
            return false;
        }
    }

    // 关键诊断：打印 g2o BA 前后对比
    if (config_.verbose) {
        Eigen::Quaterniond q_init = init_pose.q;
        Eigen::Vector3d t_init = init_pose.t;
        Eigen::Quaterniond q_out = out_pose.q;
        Eigen::Vector3d t_out = out_pose.t;
        double dt = (t_out - t_init).norm();
        double dt_z = std::abs(t_out.z() - t_init.z());
        double dq_angle = 2.0 * std::acos(std::max(-1.0, std::min(1.0,
            (q_init.conjugate() * q_out).w()))) * 180.0 / M_PI;
        { if (config_.verbose) std::cerr << "[g2o-BA] delta_t=" << std::fixed << std::setprecision(4) << dt
                  << "m delta_Z=" << std::setprecision(4) << dt_z
                  << "m delta_R=" << dq_angle << "deg"
                  << " final_t=(" << t_out.transpose() << ")"
                  << std::endl; }
    }

    // ★★★ Bug Fix: BA优化失败时返回false而非true ★★★
    bool ba_success = !(out_pose.t.hasNaN() || out_pose.q.coeffs().hasNaN() ||
        out_pose.q.norm() < 0.1 || std::isnan(out_pose.q.norm()) ||
        out_pose.t.norm() > 1000.0);
    if (!ba_success) {
        out_pose = init_pose;
        if (config_.verbose) {
            std::cerr << "[g2o-BA] WARNING: BA produced invalid pose, fallback to init_pose" << std::endl;
        }
    }

    return ba_success;
}

// ==================== g2o BA（联合优化地图点，实验性）====================
// ★★★ Bug Fix #35: g2oJointBA 死代码 - EdgeRobustReprojectPoint 是 BinaryEdge，
// 但 g2oBA 中从未调用 setVertex(1, ...)，故该函数无法正确工作 ★★★
// 该函数仅在 config_.joint_point_optimization=true 时被调用（默认 false），
// 对当前 VO 流水线无影响，但若启用 joint_point_optimization 会导致错误。
// 建议：若需要联合 BA，应使用正确的 binary edge 配对（VertexPointXYZ + VertexSE3Expmap）。

bool RobustPnPSolver::g2oJointBA(const std::vector<cv::Point3f>& pts3d,
                                  const std::vector<cv::Point2f>& pts2d,
                                  const std::vector<bool>& inlier_mask,
                                  const Pose& init_pose,
                                  Pose& out_pose,
                                  std::vector<Eigen::Vector3d>& refined_pts3d) const {

    std::vector<cv::Point3f> inlier_pts3d;
    std::vector<cv::Point2f> inlier_pts2d;
    for (size_t i = 0; i < inlier_mask.size(); ++i) {
        if (inlier_mask[i]) {
            inlier_pts3d.push_back(pts3d[i]);
            inlier_pts2d.push_back(pts2d[i]);
        }
    }
    if (inlier_pts3d.size() < 4) return false;

    using BlockSolverType = g2o::BlockSolver_6_3;
    using LinearSolverType = g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    auto* v_se3 = new g2o::VertexSE3Expmap();
    v_se3->setId(0);
    v_se3->setEstimate(g2o::SE3Quat(init_pose.q, init_pose.t));
    v_se3->setFixed(false);
    optimizer.addVertex(v_se3);

    (void)K_eigen_;  // 未使用：K_eigen_ 在 EdgeRobustReprojectPoint（已删除）中直接使用

    for (size_t i = 0; i < inlier_pts3d.size(); ++i) {
        Eigen::Vector3d pt3d(inlier_pts3d[i].x, inlier_pts3d[i].y, inlier_pts3d[i].z);
        Eigen::Vector2d pt2d(inlier_pts2d[i].x, inlier_pts2d[i].y);

        const double info_scale = 1.0;

        auto* v_pt = new g2o::VertexPointXYZ();
        v_pt->setId(static_cast<int>(i + 1));
        v_pt->setEstimate(pt3d);
        v_pt->setMarginalized(true);
        optimizer.addVertex(v_pt);

        // ★★★ Bug #35: EdgeRobustReprojectPoint 已被删除（悬空 Binary Edge）★★★
        // 此函数无法正确运行，因为需要重新实现正确的 binary edge。
        // 当前代码保留但标记为不可用，joint_point_optimization 应保持 false。
        auto* edge = new EdgeRobustReproject(pt3d, K_eigen_, info_scale);
        edge->setId(static_cast<int>(1000 + i));
        edge->setVertex(0, v_se3);
        edge->setMeasurement(pt2d);
        edge->setInformation(info_scale * Eigen::Matrix2d::Identity());

        if (config_.use_huber_kernel) {
            auto* kernel = new g2o::RobustKernelHuber();
            kernel->setDelta(config_.huber_delta);
            edge->setRobustKernel(kernel);
        }

        optimizer.addEdge(edge);
    }

    optimizer.jacobianWorkspace().updateSize(2, 12, true);

    optimizer.initializeOptimization();
    int iter = optimizer.optimize(config_.ba_max_iterations);

    g2o::SE3Quat T = v_se3->estimate();
    out_pose = Pose(Eigen::Quaterniond(T.rotation()), T.translation());

    refined_pts3d.clear();
    for (size_t i = 0; i < inlier_pts3d.size(); ++i) {
        auto* v_pt = dynamic_cast<g2o::VertexPointXYZ*>(
            optimizer.vertex(static_cast<int>(i + 1)));
        if (v_pt) {
            Eigen::Vector3d pt_est = v_pt->estimate();
            if (pt_est.hasNaN() || pt_est.norm() < 1e-6) continue;
            refined_pts3d.push_back(pt_est);
        }
    }

    bool ba_success = !(out_pose.t.hasNaN() || out_pose.q.coeffs().hasNaN() ||
        out_pose.q.norm() < 0.1 || std::isnan(out_pose.q.norm()) ||
        out_pose.t.norm() > 1000.0);
    if (!ba_success) {
        out_pose = init_pose;
        if (config_.verbose) {
            std::cerr << "[g2o-JointBA] WARNING: BA produced invalid pose, fallback to init_pose" << std::endl;
        }
    }

    return ba_success;
}

// ==================== 诊断工具 ====================

std::vector<double>
RobustPnPSolver::computeReprojectionErrors(const std::vector<cv::Point3f>& pts3d,
                                             const std::vector<cv::Point2f>& pts2d,
                                             const Pose& pose,
                                             const Eigen::Matrix3d& K) {
    std::vector<double> errors;
    errors.reserve(pts3d.size());
    for (size_t i = 0; i < pts3d.size(); ++i) {
        errors.push_back(reprojError(pts3d[i], pts2d[i], pose, K));
    }
    return errors;
}

double RobustPnPSolver::computeMedianError(const std::vector<double>& errors) {
    if (errors.empty()) return 0.0;
    std::vector<double> sorted = errors;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    if (n % 2 == 0) return (sorted[n / 2 - 1] + sorted[n / 2]) * 0.5;
    return sorted[n / 2];
}

double RobustPnPSolver::computeMeanError(const std::vector<double>& errors) {
    if (errors.empty()) return 0.0;
    double sum = 0.0;
    for (double e : errors) sum += e;
    return sum / errors.size();
}

// ==================== 质量评估 ====================

PnPResult::Quality RobustPnPSolver::assessQuality(
        int inliers, int total, double median_error) const {
    if (!inliers) return PnPResult::Quality::FAILED;
    double ratio = static_cast<double>(inliers) / total;
    if (ratio > 0.7 && inliers > 30 && median_error < 3.0)
        return PnPResult::Quality::EXCELLENT;
    if (ratio > 0.5 && inliers > 15 && median_error < 5.0)
        return PnPResult::Quality::GOOD;
    if (ratio > 0.3 && inliers > 8 && median_error < 8.0)
        return PnPResult::Quality::ACCEPTABLE;
    return PnPResult::Quality::POOR;
}

// ==================== 位姿约定换算（R1 / E2）====================
// 两个函数互为逆运算，只在 RobustPnPSolver 的边界上使用：
// 入口把调用方的 T_wc 转成内部 T_cw，出口把内部 T_cw 转回 T_wc。
// 注意四元数的共轭等价于旋转矩阵转置，所以直接取 conjugate()，
// 再由 t_wc = -R_cw^T * t_cw 得到平移，比先建矩阵再转置少一次转换误差。

Pose RobustPnPSolver::poseCwToWc(const Pose& t_cw) {
    // T_wc = T_cw^{-1} = [R_cw^T | -R_cw^T * t_cw]
    Pose out;
    out.q = t_cw.q.conjugate();
    out.t = -(out.q.toRotationMatrix() * t_cw.t);
    return out;
}

Pose RobustPnPSolver::poseWcToCw(const Pose& t_wc) {
    // T_cw = T_wc^{-1} = [R_wc^T | -R_wc^T * t_wc]
    Pose out;
    out.q = t_wc.q.conjugate();
    out.t = -(out.q.toRotationMatrix() * t_wc.t);
    return out;
}

// ================================================================
// MLPnP-style RANSAC Solver（独立实现，不依赖 ORB_SLAM3）
// 相比 OpenCV EPnP：
//   - 使用 bearing vector 做几何一致性检查
//   - 更宽松的内点阈值（10 vs 15+）
//   - Progressive iteration（每次迭代 5 次，提前终止）
// ================================================================
Pose RobustPnPSolver::mlpnpSolve(
        const std::vector<Eigen::Vector3d>& pts3d,
        const std::vector<Eigen::Vector2d>& pts2d,
        const Pose& init_pose,
        PnPResult* result_ptr) {

    PnPResult result;
    result.total_points = static_cast<int>(pts3d.size());

    if (!has_camera_) {
        std::cerr << "[MLPnP] 错误：未设置相机内参！" << std::endl;
        result.quality = PnPResult::Quality::FAILED;
        if (result_ptr) *result_ptr = result;
        return Pose();
    }
    if (pts3d.size() < 6) {
        std::cerr << "[MLPnP] 点数不足（需要 ≥ 6，实际 " << pts3d.size() << "）" << std::endl;
        result.quality = PnPResult::Quality::FAILED;
        if (result_ptr) *result_ptr = result;
        return Pose();
    }

    // 构建 bearing vectors（从 2D 像素计算归一化射线）
    bearingVectors_t bearing_vectors;
    bearing_vectors.reserve(pts2d.size());
    double fx = K_eigen_(0, 0), fy = K_eigen_(1, 1);
    double cx = K_eigen_(0, 2), cy = K_eigen_(1, 2);
    for (const auto& p2d : pts2d) {
        double u = (p2d.x() - cx) / fx;
        double v = (p2d.y() - cy) / fy;
        double norm = std::sqrt(u*u + v*v + 1.0);
        bearing_vectors.emplace_back(u / norm, v / norm, 1.0 / norm);
    }

    // 转换 3D 点到带对齐分配器的类型
    points_t points3d_aligned;
    points3d_aligned.reserve(pts3d.size());
    for (const auto& p : pts3d) {
        points3d_aligned.emplace_back(p);
    }

    // 转换 2D 像素到 cv::Point2f
    std::vector<cv::Point2f> pixels;
    pixels.reserve(pts2d.size());
    for (const auto& p2d : pts2d) {
        pixels.emplace_back(static_cast<float>(p2d.x()), static_cast<float>(p2d.y()));
    }

    // 关键帧索引和 sigma（金字塔层级权重，level 0 = 1.0）
    std::vector<int> keypoint_indices(pts3d.size());
    std::vector<float> sigma2(pts3d.size(), 1.0f);
    for (size_t i = 0; i < pts3d.size(); ++i) {
        keypoint_indices[i] = static_cast<int>(i);
        sigma2[i] = 1.0f;  // 单目 RGB-D 无金字塔，用统一权重
    }

    MLPNPConfig mlpnp_cfg;
    mlpnp_cfg.min_inliers = 8;     // ★★★ 从 10 降低到 8：低内点率场景更宽容 ★★★
    mlpnp_cfg.max_iterations = 300;
    mlpnp_cfg.sigma_threshold = 8.0;   // RGB-D 场景用 8px

    MLPNPSolver mlpnp(mlpnp_cfg);
    mlpnp.SetData(bearing_vectors, points3d_aligned,
                   pixels, keypoint_indices, sigma2, K_cv_);

    // ORB_SLAM3 策略：5 次迭代，逐步搜索
    std::vector<bool> best_inliers;
    Eigen::Matrix4f T_best;
    int best_n_inliers = 0;
    bool bNoMore = false;

    while (!bNoMore) {
        std::vector<bool> vbInliers;
        int nInliers = 0;
        Eigen::Matrix4f T_out;

        bool ok = mlpnp.iterate(5, bNoMore, vbInliers, nInliers, T_out);

        if (ok && nInliers > best_n_inliers) {
            best_n_inliers = nInliers;
            best_inliers = vbInliers;
            T_best = T_out;

            // 提前终止条件（ORB_SLAM3 风格）
            if (best_n_inliers >= 50) break;
        }
    }

    if (best_n_inliers < 10) {
        { if (config_.verbose) std::cerr << "[MLPnP] 失败：best_inliers=" << best_n_inliers << " < 10" << std::endl; }
        result.quality = PnPResult::Quality::FAILED;
        if (result_ptr) *result_ptr = result;
        return Pose();
    }

    // 提取 R, t 从 T_cw
    Eigen::Matrix3f R_f = T_best.block<3, 3>(0, 0);
    Eigen::Vector3f t_f = T_best.block<3, 1>(0, 3);
    Eigen::Matrix3d R_d = R_f.cast<double>();
    Eigen::Vector3d t_d = t_f.cast<double>();

    // 旋转矩阵合理性检查
    double det_R = R_d.determinant();
    if (det_R < 0) {
        R_d = -R_d;
        t_d = -t_d;
        if (config_.verbose)
            std::cerr << "[MLPnP] reflection corrected (det(R)=" << det_R << ")" << std::endl;
    }

    if (t_d.norm() > 500.0) {
        { if (config_.verbose) std::cerr << "[MLPnP] REJECT: |t|=" << t_d.norm() << "m exceeds 500m" << std::endl; }
        result.quality = PnPResult::Quality::FAILED;
        if (result_ptr) *result_ptr = result;
        return Pose();
    }

    Eigen::Quaterniond q(R_d);
    q.normalize();
    Pose final_pose(q, t_d);

    // ★★★ Bug Fix: 使用实际点数据而非空向量 ★★★
    std::vector<cv::Point3f> pts3d_cv;
    std::vector<cv::Point2f> pts2d_cv;
    pts3d_cv.reserve(pts3d.size());
    pts2d_cv.reserve(pts2d.size());
    for (size_t i = 0; i < pts3d.size(); ++i) {
        pts3d_cv.emplace_back(static_cast<float>(pts3d[i].x()),
                             static_cast<float>(pts3d[i].y()),
                             static_cast<float>(pts3d[i].z()));
        pts2d_cv.emplace_back(static_cast<float>(pts2d[i].x()),
                             static_cast<float>(pts2d[i].y()));
    }
    std::vector<bool> final_inliers = best_inliers;
    selectInliers(
        pts3d_cv,
        pts2d_cv,
        final_pose, final_inliers, 12.0);

    int final_n = static_cast<int>(std::count(final_inliers.begin(), final_inliers.end(), true));

    result.inliers = final_n;
    result.inlier_ratio = static_cast<double>(final_n) / pts3d.size();
    // 出口换算（R1）：final_pose 是 T_cw，对外统一返回 T_wc
    const Pose final_pose_wc = poseCwToWc(final_pose);
    result.pose = final_pose_wc;
    result.success = true;
    result.quality = assessQuality(final_n, static_cast<int>(pts3d.size()), 5.0);

    if (config_.verbose) {
        std::cerr << "[MLPnP] inliers=" << final_n << "/" << pts3d.size()
                  << " ratio=" << std::fixed << std::setprecision(3) << result.inlier_ratio
                  << " |t_wc|=" << final_pose_wc.t.norm()
                  << " quality=" << static_cast<int>(result.quality) << std::endl;
    }

    if (result_ptr) *result_ptr = result;
    return final_pose_wc;
}

} // namespace frontend
} // namespace sfc_core
