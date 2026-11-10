#ifdef sfc_core
#undef sfc_core
#endif

#include "sfc_core/feature/bow_adapter.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace sfc_core {
namespace feature {

// ================================================================
// ORB 描述子 Hamming 距离
// ================================================================
inline int orbDescriptorDistance(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty()) return 256;
    return cv::norm(a, b, cv::NORM_HAMMING);
}

// ================================================================
// 辅助：将 cv::Mat (Nx32 或 Nx64, CV_8U) 转为 vector<Mat>
// 注意：DBoW2 ORBVocabulary 用 32 字节（256-bit）ORB 描述子训练，
// 但 OctreeORB 生成 64 字节（512-bit）描述子。此处统一截断到前 32 字节。
// ================================================================
static std::vector<cv::Mat> toDescriptorVector(const cv::Mat& descriptors) {
    std::vector<cv::Mat> desc_vec;
    desc_vec.reserve(descriptors.rows);
    for (int i = 0; i < descriptors.rows; ++i) {
        cv::Mat row = descriptors.row(i);
        if (row.cols > 32) {
            // 截断到前 32 字节（与 ORBVocabulary 的 L=32 对齐）
            desc_vec.push_back(row.colRange(0, 32).clone());
        } else {
            desc_vec.push_back(row.clone());
        }
    }
    return desc_vec;
}

// ================================================================
// BoWAdapter 实现
// ================================================================
BoWAdapter::BoWAdapter() {}
BoWAdapter::BoWAdapter(const BoWConfig& config) : config_(config) {
    if (!config.vocabulary_path.empty()) {
        loadVocabulary(config.vocabulary_path);
    }
}

BoWAdapter::~BoWAdapter() {}

bool BoWAdapter::loadVocabulary(const std::string& path) {
    config_.vocabulary_path = path;
    vocab_ = std::make_unique<DBoW2::ORBVocabulary>();
    bool ok = vocab_->loadFromTextFile(path);
    if (!ok) {
        vocab_.reset();
        return false;
    }
    return true;
}

bool BoWAdapter::isLoaded() const {
    return vocab_ != nullptr && !vocab_->empty();
}

void BoWAdapter::getBowVector(
    const cv::Mat& descriptor,
    std::vector<std::pair<unsigned int, double>>& bow_vec) const
{
    bow_vec.clear();
    if (!vocab_ || descriptor.empty()) return;

    auto desc_vec = toDescriptorVector(descriptor);
    DBoW2::BowVector bv;
    vocab_->transform(desc_vec, bv);

    for (const auto& kv : bv) {
        bow_vec.emplace_back(kv.first, kv.second);
    }
}

double BoWAdapter::computeScore(const cv::Mat& desc1, const cv::Mat& desc2) const {
    if (desc1.empty() || desc2.empty() || !vocab_) return 0.0;

    auto vec1 = toDescriptorVector(desc1);
    auto vec2 = toDescriptorVector(desc2);

    DBoW2::BowVector bv1, bv2;
    vocab_->transform(vec1, bv1);
    vocab_->transform(vec2, bv2);

    return vocab_->score(bv1, bv2);
}

void BoWAdapter::addKeyFrame(int frame_id, const cv::Mat& descriptors) {
    if (!vocab_ || descriptors.empty()) return;

    auto desc_vec = toDescriptorVector(descriptors);  // 截断到 32 字节（与 Vocabulary 对齐）
    DBoW2::BowVector bow_vec;
    DBoW2::FeatureVector feat_vec;
    vocab_->transform(desc_vec, bow_vec, feat_vec, 0);

    FrameCacheEntry entry;
    // ★★★ Bug Fix: 存储截断后的 32 字节描述子，与 searchByBoW 中的截断版本一致 ★★★
    // 原 Bug: 存储原始 64 字节描述子，导致 searchByBoW 中
    // orbDescriptorDistance(64字节kf_desc, 32字节截断帧_desc) 计算错位
    entry.descriptors = cv::Mat();
    entry.descriptors.create(descriptors.rows, 32, CV_8U);
    for (int i = 0; i < descriptors.rows; ++i) {
        cv::Mat src_row = descriptors.row(i);
        cv::Mat dst_row = entry.descriptors.row(i);
        if (src_row.cols > 32) {
            src_row.colRange(0, 32).copyTo(dst_row);
        } else {
            src_row.copyTo(dst_row);
        }
    }
    entry.bow_vec = std::move(bow_vec);
    entry.feat_vec = std::move(feat_vec);
    registered_frames_[frame_id] = std::move(entry);
}

std::vector<std::pair<int, double>> BoWAdapter::queryCandidates(
    const cv::Mat& descriptors,
    double min_score) const
{
    std::vector<std::pair<int, double>> results;
    if (!vocab_ || descriptors.empty() || registered_frames_.empty()) {
        return results;
    }

    auto desc_vec = toDescriptorVector(descriptors);
    DBoW2::BowVector query_bow;
    vocab_->transform(desc_vec, query_bow);

    for (const auto& kv : registered_frames_) {
        int fid = kv.first;
        const DBoW2::BowVector& ref_bow = kv.second.bow_vec;
        double s = vocab_->score(query_bow, ref_bow);
        if (s > min_score) {
            results.emplace_back(fid, s);
        }
    }

    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    return results;
}

// ================================================================
// SearchByBoW — 核心：ORB-SLAM3 风格几何 BoW 匹配
//
// 算法：
// 1. 对两帧分别用同一 Vocabulary 量化，获取 FeatureVector
//   FeatureVector: map<NodeId, vector<feature_idx>>
//   NodeId = 词汇树某层的节点，同一 NodeId 下的特征具有相似视觉外观
// 2. 在同一 NodeId 下，用 Hamming 距离做描述子匹配 + Lowe's ratio test
// 3. 匹配成功的特征间接对应到地图点（KF 有 MapPoint 信息）
//
// ORB-SLAM3 ORBmatcher.cc:223 SearchByBoW()
// ================================================================
int BoWAdapter::searchByBoW(
    const cv::Mat& frame_descriptors,
    int ref_kf_id,
    std::vector<int>& matches) const
{
    matches.clear();
    if (!vocab_ || frame_descriptors.empty()) return 0;

    auto it = registered_frames_.find(ref_kf_id);
    if (it == registered_frames_.end()) return 0;

    const FrameCacheEntry& kf_entry = it->second;
    const DBoW2::FeatureVector& kf_feat_vec = kf_entry.feat_vec;

    // 计算当前帧的 FeatureVector（levelsup=0 即叶子层 word）
    auto frame_desc_vec = toDescriptorVector(frame_descriptors);
    DBoW2::BowVector frame_bow_vec;
    DBoW2::FeatureVector frame_feat_vec;
    vocab_->transform(frame_desc_vec, frame_bow_vec, frame_feat_vec, 0);

    const cv::Mat& kf_descriptors = kf_entry.descriptors;

    matches.assign(frame_descriptors.rows, -1);

    int nmatches = 0;
    const int TH_LOW = 50;
    const float nn_ratio = 0.6f;

    // 同时遍历两帧的 FeatureVector（都按 NodeId 排序）
    auto kf_it = kf_feat_vec.begin();
    auto frame_it = frame_feat_vec.begin();
    const auto kf_end = kf_feat_vec.end();
    const auto frame_end = frame_feat_vec.end();

    while (kf_it != kf_end && frame_it != frame_end) {
        if (kf_it->first == frame_it->first) {
            // 同一 NodeId 下的特征候选
            const std::vector<unsigned int>& kf_indices = kf_it->second;
            const std::vector<unsigned int>& frame_indices = frame_it->second;

            for (unsigned int i_kf : kf_indices) {
                const cv::Mat& desc_kf = kf_descriptors.row(i_kf);

                int best_dist1 = 256;
                int best_idx_f = -1;
                int best_dist2 = 256;

                for (unsigned int i_f : frame_indices) {
                    if (matches[i_f] >= 0) continue;  // 已匹配

                    const cv::Mat& desc_f = frame_descriptors.row(i_f);
                    int dist = orbDescriptorDistance(desc_kf, desc_f);

                    if (dist < best_dist1) {
                        best_dist2 = best_dist1;
                        best_dist1 = dist;
                        best_idx_f = (int)i_f;
                    } else if (dist < best_dist2) {
                        best_dist2 = dist;
                    }
                }

                // Lowe's ratio test
                if (best_dist1 <= TH_LOW) {
                    if (static_cast<float>(best_dist1) < nn_ratio * static_cast<float>(best_dist2)) {
                        matches[best_idx_f] = (int)i_kf;
                        ++nmatches;
                    }
                }
            }

            ++kf_it;
            ++frame_it;
        } else if (kf_it->first < frame_it->first) {
            ++kf_it;
        } else {
            ++frame_it;
        }
    }

    return nmatches;
}

} // namespace feature
} // namespace sfc_core
