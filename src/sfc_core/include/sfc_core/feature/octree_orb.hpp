#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_FEATURE_OCTREE_ORB_HPP_
#define SFC_CORE_FEATURE_OCTREE_ORB_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <list>

namespace sfc_core {
namespace feature {

// ================================================================
// Octree ORB 配置（完全对齐 ORB_SLAM3 ORBextractor）
// ================================================================
struct OctreeORBConfig {
    int n_features = 2000;          // 每层金字塔提取的特征总数
    float scale_factor = 1.2f;       // 金字塔层间尺度因子
    int n_levels = 8;                // 金字塔层数
    int ini_threshold_fast = 20;      // 初始 FAST 阈值
    int min_threshold_fast = 7;      // 最小 FAST 阈值
    int edge_threshold = 19;         // 边缘阈值（ORB_SLAM3 标准值）
    int patch_size = 31;             // 描述子 patch 大小（必须为奇数）
};

// ================================================================
// ExtractorNode — 四叉树节点（与 ORB_SLAM3 完全一致）
// ================================================================
class ExtractorNode {
public:
    ExtractorNode() : b_no_more(false) {}

    void divideNode(ExtractorNode& n1, ExtractorNode& n2,
                    ExtractorNode& n3, ExtractorNode& n4);

    std::vector<cv::KeyPoint> v_keys;
    cv::Point2i ul, ur, bl, br;
    std::list<ExtractorNode>::iterator lit;
    bool b_no_more;
};

// ================================================================
// OctreeORB — 与 ORB_SLAM3 ORBextractor 行为完全对齐
// ================================================================
class OctreeORB {
public:
    explicit OctreeORB(const OctreeORBConfig& config);

    void extract(const cv::Mat& img,
                 const cv::Mat& img_right,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors);

    const std::vector<float>& getScaleFactors() const { return mv_scale_factor_; }
    const std::vector<float>& getInvScaleFactors() const { return mv_inv_scale_factor_; }
    const std::vector<float>& getLevelSigma2() const { return mv_level_sigma2_; }
    const std::vector<float>& getInvLevelSigma2() const { return mv_inv_level_sigma2_; }
    int getNumLevels() const { return config_.n_levels; }

    // ORB 描述子距离（Hamming）
    static int descriptorDistance(const cv::Mat& a, const cv::Mat& b);

private:
    void computePyramid(const cv::Mat& img);
    void computeKeyPointsOctTree(
        std::vector<std::vector<cv::KeyPoint>>& all_keypoints);
    void distributeOctTree(
        const std::vector<cv::KeyPoint>& v_to_distribute_keys,
        const int& min_x, const int& max_x,
        const int& min_y, const int& max_y,
        const int& n_features, const int& level,
        std::vector<cv::KeyPoint>& result);

    OctreeORBConfig config_;
    std::vector<cv::Mat> image_pyramid_;
    std::vector<float> mv_scale_factor_;
    std::vector<float> mv_inv_scale_factor_;
    std::vector<float> mv_level_sigma2_;
    std::vector<float> mv_inv_level_sigma2_;
    std::vector<int> mn_features_per_level_;
    std::vector<cv::Point> pattern_;
    std::vector<int> umax_;
};

} // namespace feature
} // namespace sfc_core

#endif // SFC_CORE_FEATURE_OCTREE_ORB_HPP_
