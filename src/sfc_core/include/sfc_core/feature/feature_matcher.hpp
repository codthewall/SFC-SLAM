// 必须在最开头清除 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_FEATURE_FEATURE_MATCHER_HPP_
#define SFC_CORE_FEATURE_FEATURE_MATCHER_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <Eigen/Dense>

namespace sfc_core {

// 前向声明（用于 SearchByProjection）
namespace frontend {
class Frame;
}

namespace feature {

// ================================================================
// Matcher 配置
// ================================================================
struct MatcherConfig {
    bool cross_check = false;       // 是否做十字交叉校验
    float lowe_ratio = 0.6f;       // Lowe's ratio 阈值
    int max_hamming_dist = 256;     // 最大汉明距离阈值
    int descriptor_dim = 64;        // ORB 描述子维度（64 bytes = 512 bits，OctreeORB 使用 512 点）

    // ================================================================
    // SearchByProjection 配置
    // ================================================================
    int th_low = 50;               // ORB_SLAM3 TH_LOW
    int th_high = 100;              // ORB_SLAM3 TH_HIGH
    int histo_length = 30;          // 方向直方图长度
    bool check_orientation = true;   // 是否检查方向一致性
};

// ================================================================
// FeatureMatcher — ORB 特征匹配器
// 支持：Hamming 距离匹配 + Lowe's ratio 过滤 + 几何一致性过滤
// ================================================================
class FeatureMatcher {
public:
    explicit FeatureMatcher(const MatcherConfig& config);

    // 用 BFMatcher 做 ORB 描述子匹配
    // descriptors1: query（历史帧），descriptors2: train（当前帧）
    void match(const cv::Mat& descriptors1,
               const cv::Mat& descriptors2,
               std::vector<cv::DMatch>& matches);

    // 对匹配结果做几何一致性过滤（基于极线约束）
    // keypoints1: query 特征点，keypoints2: train 特征点
    // F_matrix: 基础矩阵（可选，3x3 CV_64F）
    void removeOutliers(const std::vector<cv::DMatch>& raw_matches,
                        const std::vector<cv::KeyPoint>& keypoints1,
                        const std::vector<cv::KeyPoint>& keypoints2,
                        std::vector<cv::DMatch>& inlier_matches,
                        const cv::Mat& F_matrix = cv::Mat());

    // ================================================================
    // ORB_SLAM3 风格 SearchByProjection
    // 将局部地图点投影到当前帧，在投影点附近搜索匹配
    // 返回匹配数量
    // ================================================================
    // 地图点数据结构（与 ORB_SLAM3 MapPoint 对应）
    struct MapPointMatch {
        Eigen::Vector3d world_point;           // 世界坐标
        cv::Mat descriptor;                   // 最佳描述子
        int observations = 0;                  // 被观测次数
        bool is_bad = false;                   // 是否被标记为坏点

        // 投影相关（由 Tracking 填充）
        bool track_in_view = false;            // 是否在视野内
        float track_proj_x = 0, track_proj_y = 0;  // 投影像素坐标
        float track_view_cos = 0;              // 视角余弦
        int track_scale_level = -1;             // 金字塔层级
        float track_depth = 0;                 // 深度
    };

    // SearchByProjection: 帧到帧的投影匹配
    // last_pts3d: 上一帧的 3D 点（相机坐标系）
    // curr_kpts: 当前帧特征点
    // curr_descriptors: 当前帧描述子
    // R, t: last 到 curr 的相对变换
    // K: 相机内参
    // 返回: 匹配对 vector<pair<int last_idx, int curr_idx>>
    std::vector<std::pair<int, int>> searchByProjection(
        const std::vector<Eigen::Vector3d>& last_pts3d,
        const std::vector<Eigen::Vector3d>& last_pts3d_world,
        const std::vector<bool>& last_depth_valid,
        const std::vector<cv::KeyPoint>& curr_kpts,
        const cv::Mat& curr_descriptors,
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& t,
        const cv::Mat& K,
        float th = 3.0f,
        bool check_scale = true);

    // 投影地图点到当前帧（计算 track_in_view 等成员）
    // 返回在视野内的点数
    int projectMapPointsToFrame(
        const std::vector<MapPointMatch*>& map_points,
        const Eigen::Matrix3d& R_cw,
        const Eigen::Vector3d& t_cw,
        const cv::Mat& K,
        float fx, float fy, float cx, float cy,
        int cols, int rows,
        float th = 3.0f);

    // 获取配置
    const MatcherConfig& config() const { return config_; }

private:
    // 根据视角余弦计算搜索半径
    static float radiusByViewingCos(float viewCos);

    // 计算直方图中最大的三个峰值
    static void computeThreeMaxima(std::vector<int>* histo, int L, int& ind1, int& ind2, int& ind3);

    MatcherConfig config_;
};

} // namespace feature
} // namespace sfc_core

#endif // SFC_CORE_FEATURE_FEATURE_MATCHER_HPP_
