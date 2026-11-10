// 必须在最开头清除 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_FEATURE_BOW_ADAPTER_HPP_
#define SFC_CORE_FEATURE_BOW_ADAPTER_HPP_

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>
#include <opencv2/opencv.hpp>
#include "sfc_core/thirdparty/dbow2/ORBVocabulary.h"

namespace sfc_core {
namespace feature {

// ================================================================
// BoW 配置
// ================================================================
struct BoWConfig {
    std::string vocabulary_path;
    int voc_branching_factor = 10;
    int voc_depth_levels = 6;
    bool use_idf = true;
    int min_vocab_score = 0;
};

// ================================================================
// BoWAdapter — 使用真实 DBoW2 ORBVocabulary
// ================================================================
class BoWAdapter {
public:
    explicit BoWAdapter();
    explicit BoWAdapter(const BoWConfig& config);
    ~BoWAdapter();

    bool loadVocabulary(const std::string& path);
    bool isLoaded() const;

    // 获取描述子对应的 BoW 向量
    void getBowVector(const cv::Mat& descriptor,
                      std::vector<std::pair<unsigned int, double>>& bow_vec) const;

    // 计算两帧 BoW 相似度（L1）
    double computeScore(const cv::Mat& desc1, const cv::Mat& desc2) const;

    // 注册关键帧
    void addKeyFrame(int frame_id, const cv::Mat& descriptors);

    // 查询候选帧
    std::vector<std::pair<int, double>> queryCandidates(
        const cv::Mat& descriptors,
        double min_score = 0.03) const;

    // ========== SearchByBoW 核心接口 ==========
    // 对齐 ORB-SLAM3 的 SearchByBoW 功能
    // 返回：frame_idx → matched_map_point_idx（-1 表示无匹配）
    // 同时输出匹配数
    int searchByBoW(const cv::Mat& frame_descriptors,
                    int ref_kf_id,
                    std::vector<int>& matches) const;

    int numRegisteredFrames() const { return (int)registered_frames_.size(); }
    const BoWConfig& config() const { return config_; }

    // 直接暴露 Vocabulary 指针（供 Frame 使用）
    const DBoW2::ORBVocabulary* vocabulary() const { return vocab_.get(); }

private:
    struct FrameCacheEntry {
        cv::Mat descriptors;
        DBoW2::BowVector bow_vec;
        DBoW2::FeatureVector feat_vec;
    };

    BoWConfig config_;
    std::unique_ptr<DBoW2::ORBVocabulary> vocab_;
    std::unordered_map<int, FrameCacheEntry> registered_frames_;
    DBoW2::BowVector last_bow_vec_;
};

} // namespace feature
} // namespace sfc_core

#endif // SFC_CORE_FEATURE_BOW_ADAPTER_HPP_
