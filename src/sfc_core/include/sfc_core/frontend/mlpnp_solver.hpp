// ================================================================
//  MLPnPsolver - 基于 Bearing Vector 的 PnP 求解器
//  移植自 ORB_SLAM3，使用最小 6 点法 + 高斯-牛顿优化
//  相比 OpenCV EPnP：对退化场景（平面、近似平面）更鲁棒
// ================================================================

#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_FRONTEND_MLPNP_SOLVER_HPP_
#define SFC_CORE_FRONTEND_MLPNP_SOLVER_HPP_

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>

namespace sfc_core {
namespace frontend {

// ==================== 类型定义 ====================

/** 单位向量，描述相机帧中的观测方向（始终在相机坐标系下表达） */
typedef Eigen::Vector3d bearingVector_t;

/** 单位向量数组 */
typedef std::vector<bearingVector_t, Eigen::aligned_allocator<bearingVector_t>> bearingVectors_t;

/** 2D 协方差矩阵（bearing vector 的 2D 不确定性） */
typedef Eigen::Matrix2d cov2_mat_t;

/** 3D 协方差矩阵 */
typedef Eigen::Matrix3d cov3_mat_t;

/** 3D 协方差矩阵数组 */
typedef std::vector<cov3_mat_t, Eigen::aligned_allocator<cov3_mat_t>> cov3_mats_t;

/** 3D 点 */
typedef Eigen::Vector3d point_t;

/** 3D 点数组 */
typedef std::vector<point_t, Eigen::aligned_allocator<point_t>> points_t;

/** 齐次 3D 点 (4-vector) */
typedef Eigen::Vector4d point4_t;

/** 齐次 3D 点数组 */
typedef std::vector<point4_t, Eigen::aligned_allocator<point4_t>> points4_t;

/** Rodrigues 参数（旋转向量） */
typedef Eigen::Vector3d rodrigues_t;

/** 旋转矩阵 */
typedef Eigen::Matrix3d rotation_t;

/** 3x4 变换矩阵 [R|t] */
typedef Eigen::Matrix<double, 3, 4> transformation_t;

/** 平移向量（相机位置） */
typedef Eigen::Vector3d translation_t;

// ==================== 配置结构 ====================

struct MLPNPConfig {
    double probability = 0.99;       // RANSAC 成功概率
    int min_inliers = 10;            // 最小内点数
    int max_iterations = 300;        // 最大 RANSAC 迭代次数
    int min_set = 6;                 // 每次迭代最小样本数
    float epsilon = 0.4;             // 期望内点比例
    float threshold = 5.991;         // 卡方检验阈值 (chi2 for 2 dof, 95%)
    float sigma_threshold = 8.0;     // 像素重投影误差阈值
};

// ==================== MLPnP 求解器类 ====================

class MLPNPSolver {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MLPNPSolver() : mnInliersi(0), mnIterations(0), mnBestInliers(0), N(0), mpCamera(nullptr) {}

    explicit MLPNPSolver(const MLPNPConfig& config) : MLPNPSolver() {
        config_ = config;
    }

    ~MLPNPSolver() = default;

    void SetRansacParameters(double probability = 0.99, int minInliers = 10,
                             int maxIterations = 300, int minSet = 6,
                             float epsilon = 0.4, float th2 = 5.991);

    void SetConfig(const MLPNPConfig& config) { config_ = config; }

    // ==================== 数据设置 ====================
    // 这些方法替代了原 ORB_SLAM3 的构造函数，允许更灵活地设置数据

    void SetData(const bearingVectors_t& bearings,
                 const points_t& points3d,
                 const std::vector<cv::Point2f>& pixels,
                 const std::vector<int>& keypoint_indices,
                 const std::vector<float>& sigma2,
                 const cv::Mat& K);

    // ==================== RANSAC 迭代 ====================
    // 返回: 成功/失败
    // bNoMore: 是否达到最大迭代次数
    // vbInliers: 内点掩码
    // nInliers: 内点数量
    // Tout: 输出变换矩阵 (4x4, float)

    bool iterate(int nIterations, bool& bNoMore,
                 std::vector<bool>& vbInliers, int& nInliers,
                 Eigen::Matrix4f& Tout);

    // ==================== 便捷接口 ====================

    bool solve(const points_t& points3d,
               const bearingVectors_t& bearings,
               const std::vector<cv::Point2f>& pixels,
               Eigen::Matrix4f& output_pose,
               std::vector<bool>& inliers,
               int& n_inliers);

private:
    // ==================== 内部方法 ====================

    void CheckInliers();
    bool Refine();

    /**
     * 计算相机位姿
     * @param f bearing vectors
     * @param p 3D 点
     * @param covMats 协方差矩阵数组（可选，当前版本不使用）
     * @param indices 用于计算的点索引
     * @param result 输出变换矩阵 [R|t]
     */
    void computePose(const bearingVectors_t& f,
                     const points_t& p,
                     const cov3_mats_t& covMats,
                     const std::vector<int>& indices,
                     transformation_t& result);

    /**
     * 高斯-牛顿优化
     */
    void mlpnp_gn(Eigen::VectorXd& x,
                  const points_t& pts,
                  const std::vector<Eigen::MatrixXd>& nullspaces,
                  const Eigen::SparseMatrix<double> Kll,
                  bool use_cov);

    /**
     * 计算残差和雅可比矩阵
     */
    void mlpnp_residuals_and_jacs(const Eigen::VectorXd& x,
                                  const points_t& pts,
                                  const std::vector<Eigen::MatrixXd>& nullspaces,
                                  Eigen::VectorXd& r,
                                  Eigen::MatrixXd& fjac,
                                  bool getJacs);

    /**
     * MLPnP 雅可比矩阵计算
     */
    void mlpnpJacs(const point_t& pt,
                    const Eigen::Vector3d& nullspace_r,
                    const Eigen::Vector3d& nullspace_s,
                    const rodrigues_t& w,
                    const translation_t& t,
                    Eigen::MatrixXd& jacs);

    // ==================== Rodrigues 转换 ====================

    static Eigen::Matrix3d rodrigues2rot(const Eigen::Vector3d& omega);
    static Eigen::Vector3d rot2rodrigues(const Eigen::Matrix3d& R);

    // ==================== 随机数 ====================

    static int randomInt(int max) {
        return rand() % max;
    }

    // ==================== 数据成员 ====================

    MLPNPConfig config_;

    // 匹配的地图片
    std::vector<void*> mvpMapPointMatches;

    // 2D 像素坐标
    std::vector<cv::Point2f> mvP2D;

    // Bearing vectors (归一化射线方向)
    bearingVectors_t mvBearingVecs;

    // 尺度层级权重
    std::vector<float> mvSigma2;

    // 内点误差阈值（RANSAC 用）
    std::vector<float> mvMaxError;

    // 3D 点 (世界坐标系)
    points_t mvP3Dw;

    // 关键点在帧中的索引
    std::vector<size_t> mvKeyPointIndices;

    // 当前估计
    double mRi[3][3];
    double mti[3];
    Eigen::Matrix4f mTcwi;
    std::vector<bool> mvbInliersi;
    int mnInliersi;

    // 当前 RANSAC 状态
    int mnIterations;
    std::vector<bool> mvbBestInliers;
    int mnBestInliers;
    Eigen::Matrix4f mBestTcw;

    // 精化后的结果
    Eigen::Matrix4f mRefinedTcw;
    std::vector<bool> mvbRefinedInliers;
    int mnRefinedInliers;

    // 匹配点数量
    int N;

    // 随机选择索引 [0 .. N-1]
    std::vector<size_t> mvAllIndices;

    // 相机内参
    cv::Mat mK;
    cv::Mat mDistCoef;
    void* mpCamera;  // 通用指针，用于兼容
};

} // namespace frontend
} // namespace sfc_core

#endif // SFC_CORE_FRONTEND_MLPNP_SOLVER_HPP_
