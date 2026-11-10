// 必须在最开头清除 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_FRONTEND_ROBUST_PNP_SOLVER_HPP_
#define SFC_CORE_FRONTEND_ROBUST_PNP_SOLVER_HPP_

#include <vector>
#include <map>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

// g2o 头文件（必须在自定义命名空间外部引入，引入后主动清除宏）
// 包括 BaseUnaryEdge, VertexSE3Expmap, SparseOptimizer,
// OptimizationAlgorithmLevenberg, BlockSolver_6_3, LinearSolverDense
#include <g2o/core/base_unary_edge.h>
#include <g2o/types/sba/vertex_se3_expmap.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/block_solver.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include "sfc_core/frontend/frame.hpp"
#include "sfc_core/frontend/mlpnp_solver.hpp"

namespace sfc_core {
namespace frontend {

// ==================== g2o 边：重投影误差（相机位姿顶点 → 虚拟测量）====================
// 此类需要在 hpp 中声明（而非仅在 cpp 中），因为 header 成员 ba_edges_ 持有其指针。
class EdgeRobustReproject : public g2o::BaseUnaryEdge<2, Eigen::Vector2d,
                                                        g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeRobustReproject() {
        _pt3d.setZero();
        _K.setIdentity();
        _info_scale = 1.0;
        _measurement.setZero();
        _information.setIdentity();
    }

    EdgeRobustReproject(const Eigen::Vector3d& pt3d,
                        const Eigen::Matrix3d& K,
                        double info_scale = 1.0)
        : _pt3d(pt3d), _K(K), _info_scale(info_scale) {}

    void computeError() override {
        const g2o::VertexSE3Expmap* v_se3 =
            static_cast<const g2o::VertexSE3Expmap*>(_vertices[0]);
        g2o::SE3Quat T = v_se3->estimate();
        Eigen::Vector3d xyz_cam = T.map(_pt3d);
        double x = xyz_cam(0), y = xyz_cam(1), z = xyz_cam(2);

        // 深度太浅（< 5cm）或 NaN/Inf → 零残差
        if (z < 0.05 || !xyz_cam.allFinite()) {
            _error(0) = 1e6;
            _error(1) = 1e6;
            return;
        }

        double invz = 1.0 / z;
        // 溢出保护：若 z 极小导致 invz 溢出，置零残差
        if (!std::isfinite(invz)) {
            _error(0) = 1e6;
            _error(1) = 1e6;
            return;
        }

        _error(0) = _measurement(0) - (_K(0, 0) * x * invz + _K(0, 2));
        _error(1) = _measurement(1) - (_K(1, 1) * y * invz + _K(1, 2));
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

    void setPoint(const Eigen::Vector3d& pt3d) { _pt3d = pt3d; }
    void setCalibration(const Eigen::Matrix3d& K) { _K = K; }
    void setInfoScale(double s) { _info_scale = s; }

private:
    Eigen::Vector3d _pt3d;
    Eigen::Matrix3d _K;
    double _info_scale = 1.0;
};

struct PnPResult {
    bool success = false;
    Pose pose;

    int total_points = 0;
    int inliers = 0;
    double inlier_ratio = 0.0;
    double median_reproj_error = 0.0;
    double mean_reproj_error = 0.0;
    double final_threshold = 0.0;
    int ransac_iterations_used = 0;

    enum class Quality {
        EXCELLENT,
        GOOD,
        ACCEPTABLE,
        POOR,
        FAILED
    };
    Quality quality = Quality::FAILED;
};

struct RobustPnPConfig {
    int max_ransac_iterations = 200;
    double initial_inlier_thresh_px = 8.0;
    double final_inlier_thresh_px = 4.0;
    double confidence = 0.999;
    bool use_adaptive_threshold = true;
    double min_inlier_ratio = 0.05;

    int expansion_rounds = 2;
    double expansion_factor = 1.5;

    int ba_max_iterations = 15;
    bool use_huber_kernel = true;
    double huber_delta = 2.45;  // 2.45 = sqrt(5.991)，与 ORB_SLAM3 Optimizer.cc:852 一致
    bool use_depth_weighting = false;  // 已废弃，BA 中不再按深度加权
    bool joint_point_optimization = false;

    bool verbose = false;
};

class RobustPnPSolver {
public:
    explicit RobustPnPSolver(const RobustPnPConfig& config = RobustPnPConfig());
    explicit RobustPnPSolver(const cv::Mat& K, const RobustPnPConfig& config = RobustPnPConfig());

    void setCamera(const cv::Mat& K, const cv::Mat& D = cv::Mat());
    void setCamera(const Eigen::Matrix3d& K_eigen);

    Pose optimize(const std::vector<cv::Point3f>& pts3d,
                  const std::vector<cv::Point2f>& pts2d,
                  const Pose& init_pose,
                  PnPResult* result = nullptr,
                  bool use_motion_prior = false);

    Pose optimize(const std::vector<Eigen::Vector3d>& pts3d,
                  const std::vector<Eigen::Vector2d>& pts2d,
                  const Pose& init_pose,
                  PnPResult* result = nullptr,
                  bool use_motion_prior = false);

    static std::vector<double>
    computeReprojectionErrors(const std::vector<cv::Point3f>& pts3d,
                             const std::vector<cv::Point2f>& pts2d,
                             const Pose& pose,
                             const Eigen::Matrix3d& K);

    static double computeMedianError(const std::vector<double>& errors);
    static double computeMeanError(const std::vector<double>& errors);

    // ==================== 位姿约定（R1 / E2）====================
    // 全系统约定：Pose{q,t} == T_wc（P_world = R_wc * P_cam + t_wc）。
    // OpenCV PnP / g2o VertexSE3Expmap 的约定：T_cw（P_cam = R_cw * P_world + t_cw）。
    // 本类内部一律 T_cw，对外一律 T_wc；转换只在本类的出入口发生一次。
    // 返回 T_wc = T_cw^{-1}：R_wc = R_cw^T，t_wc = -R_cw^T * t_cw。
    static Pose poseCwToWc(const Pose& t_cw);
    // 返回 T_cw = T_wc^{-1}：R_cw = R_wc^T，t_cw = -R_cw^T * t_wc。
    static Pose poseWcToCw(const Pose& t_wc);

    // ==================== MLPnP 求解接口 ====================
    // 使用 MLPnP 算法进行 PnP 求解（基于 bearing vector，对退化场景更鲁棒）
    Pose mlpnpSolve(const std::vector<Eigen::Vector3d>& pts3d,
                    const std::vector<Eigen::Vector2d>& pts2d,
                    const Pose& init_pose = Pose(),
                    PnPResult* result = nullptr);

    // ==================== PnP 方法选择 ====================
    // 可选：使用 MLPnP 替代 OpenCV EPnP
    enum class PnPMethod {
        EPNP,      // OpenCV EPnP (默认)
        MLPNP      // MLPnP (bearing vector based)
    };
    void setPnPMethod(PnPMethod method) { pnp_method_ = method; }
    PnPMethod getPnPMethod() const { return pnp_method_; }

private:
    bool ransacPnP(const std::vector<cv::Point3f>& pts3d,
                   const std::vector<cv::Point2f>& pts2d,
                   Pose& best_pose,
                   std::vector<bool>& inlier_mask,
                   double threshold,
                   int max_iters,
                   const Pose* init_pose = nullptr);

    int selectInliers(const std::vector<cv::Point3f>& pts3d,
                       const std::vector<cv::Point2f>& pts2d,
                       const Pose& pose,
                       std::vector<bool>& inlier_mask,
                       double threshold_px) const;

    double adaptiveThreshold(double inlier_ratio, int total_points) const;

    bool expandAndRefine(const std::vector<cv::Point3f>& pts3d,
                         const std::vector<cv::Point2f>& pts2d,
                         Pose& inout_pose,
                         std::vector<bool>& inout_inliers,
                         double& inout_thresh,
                         const Pose* init_pose = nullptr);

    bool g2oBA(const std::vector<cv::Point3f>& pts3d,
               const std::vector<cv::Point2f>& pts2d,
               const std::vector<bool>& inlier_mask,
               const Pose& init_pose,
               Pose& out_pose,
               bool use_huber_removal = false) const;

    bool g2oJointBA(const std::vector<cv::Point3f>& pts3d,
                    const std::vector<cv::Point2f>& pts2d,
                    const std::vector<bool>& inlier_mask,
                    const Pose& init_pose,
                    Pose& out_pose,
                    std::vector<Eigen::Vector3d>& refined_pts3d) const;

    PnPResult::Quality assessQuality(int inliers, int total, double median_error) const;

    RobustPnPConfig config_;
    Eigen::Matrix3d K_eigen_;
    cv::Mat K_cv_, D_cv_;
    bool has_camera_ = false;

    mutable std::vector<double> error_buffer_;

    // ==================== g2o BA 优化器复用 ====================
    // 优化：缓存g2o求解器结构，每次g2oBA调用时复用，避免重复分配内存
    mutable std::unique_ptr<g2o::SparseOptimizer> ba_optimizer_;
    mutable g2o::VertexSE3Expmap* ba_vertex_se3_ = nullptr;  // 缓存位姿顶点指针
    mutable std::vector<EdgeRobustReproject*> ba_edges_;       // 缓存边指针用于清理
    mutable std::unique_ptr<g2o::OptimizationAlgorithmLevenberg> ba_algorithm_;
    mutable std::unique_ptr<g2o::BlockSolver_6_3> ba_block_solver_;
    mutable std::unique_ptr<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>> ba_linear_solver_;

    // 初始化/重建BA求解器（内部管理缓存的optimizer）
    void initBAOptimizer(int num_edges) const;
    // 清理BA缓存的边和顶点
    void clearBAEdges() const;

    // ==================== MLPnP 求解器 ====================
    PnPMethod pnp_method_ = PnPMethod::EPNP;  // 当前使用的 PnP 方法
};

} // namespace frontend
} // namespace sfc_core

#endif // SFC_CORE_FRONTEND_ROBUST_PNP_SOLVER_HPP_
