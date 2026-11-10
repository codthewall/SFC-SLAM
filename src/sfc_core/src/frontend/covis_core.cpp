#include "sfc_core/frontend/covis_core.hpp"
#include "sfc_core/frontend/covis_manager.hpp"
#include <algorithm>
#include <climits>

namespace sfc_core {
namespace frontend {

// ============= MapPoint =============

MapPoint::MapPoint(int id, const Eigen::Vector3d& world_pt)
    : id_(id), world_pt_(world_pt) {}

void MapPoint::addObservation(const MapPointObservation& obs) {
    observations_.push_back(obs);
    observing_frames_.insert(obs.frame_id);
    if (obs_by_frame_.find(obs.frame_id) == obs_by_frame_.end()) {
        obs_by_frame_[obs.frame_id] = static_cast<int>(observations_.size()) - 1;
    }
    obs_count_++;
    last_obs_frame_id_ = obs.frame_id;
}

double MapPoint::updateReprojError(double pixel_error) {
    // 指数移动平均：近期误差权重更高
    double alpha = 0.7;
    // ★★★ 修复：首次判断逻辑 ★★★
    // 原判断 avg_reproj_error_ > 1e8 已失效（初始值改为 0 后永不触发）。
    // 改为：用 obs_count_ == 0 判断首次调用。
    if (obs_count_ == 0) {
        avg_reproj_error_ = pixel_error;  // 首次
    } else {
        avg_reproj_error_ = alpha * avg_reproj_error_ + (1 - alpha) * pixel_error;
    }
    return avg_reproj_error_;
}

bool MapPoint::passCovisibilityCheck(int min_common_frames) const {
    // 共视条件：被 ≥ min_common_frames 个不同关键帧共同观测
    // 即 observing_frames_ 集合中至少有 min_common_frames 个帧
    return static_cast<int>(observing_frames_.size()) >= min_common_frames;
}

void MapPoint::promote() {
    if (status_ != PointStatus::CANDIDATE) return;
    status_ = PointStatus::CONFIRMED;
    confirmed_frame_id_ = last_obs_frame_id_;
}

void MapPoint::discard() {
    status_ = PointStatus::DISCARDED;
}

void MapPoint::markObserved(int frame_id) {
    last_obs_frame_id_ = frame_id;
}

void MapPoint::updateWorldPoint(const Eigen::Vector3d& new_pt) {
    world_pt_ = new_pt;
}

// ★★★ 描述子计算（ORB_SLAM3 风格）★★★
void MapPoint::computeDescriptor() {
    if (observations_.empty()) return;

    // 选择 ORB 距离之和最小的描述子（与 ORB_SLAM3 MapPoint.cc 一致）
    int best_dist = INT_MAX;
    int best_idx = 0;
    for (size_t i = 0; i < observations_.size(); ++i) {
        if (observations_[i].descriptor.empty()) continue;
        int dist = 0;
        for (size_t j = 0; j < observations_.size(); ++j) {
            if (i == j || observations_[j].descriptor.empty()) continue;
            dist += descriptorDistance(observations_[i].descriptor, observations_[j].descriptor);
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    if (!observations_[best_idx].descriptor.empty()) {
        descriptor_ = observations_[best_idx].descriptor.clone();
    }
}

// ★★★ 跨平台 popcount（Linux/GCC 可直接用内联函数）★★★
#if defined(__GNUC__) || defined(__clang__)
inline int portable_popcount(uint8_t x) {
    return __builtin_popcount(x);
}
#else
inline int portable_popcount(uint8_t x) {
    // Fallback for other compilers
    int count = 0;
    while (x) { count += x & 1; x >>= 1; }
    return count;
}
#endif

// ★★★ 跨平台 ORB 描述子汉明距离 ★★★
int MapPoint::descriptorDistance(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty()) return 256;
    const uint8_t* pa = a.ptr<uint8_t>();
    const uint8_t* pb = b.ptr<uint8_t>();
    int dist = 0;
    for (int i = 0; i < 32; ++i) {
        dist += portable_popcount(pa[i] ^ pb[i]);
    }
    return dist;
}


// ============= KeyFrame =============

KeyFrame::Ptr KeyFrame::create(
        int frame_id, double timestamp,
        const std::vector<cv::KeyPoint>& kps,
        const std::vector<cv::KeyPoint>& undist_kps,
        const cv::Mat& desc,
        const std::vector<Eigen::Vector3d>& pts3d,
        const std::vector<bool>& depth_valid,
        const Pose& pose) {

    auto kf = std::make_shared<KeyFrame>();
    kf->frame_id = frame_id;
    kf->timestamp = timestamp;
    kf->pose = pose;
    kf->keypoints = kps;
    kf->undistorted_keypoints = undist_kps;
    kf->descriptors = desc.clone();
    kf->pts_3d = pts3d;
    kf->depth_valid = depth_valid;
    kf->buildSpatialIndex();
    // ★★★ 初始化缓存的旋转矩阵 ★★★
    kf->R_cached = kf->pose.q.toRotationMatrix();
    kf->R_valid_ = true;
    kf->syncRotationCache();  // 重复调用无害，保证一致性
    return kf;
}

void KeyFrame::buildSpatialIndex() {
    kpt_grid_sorted.clear();
    // ★★★ 使用去畸变后的关键点坐标构建空间格网（与重投影几何一致）★★★
    const auto& kpts =
        (!undistorted_keypoints.empty() && undistorted_keypoints.size() == keypoints.size())
            ? undistorted_keypoints
            : keypoints;
    kpt_grid_sorted.reserve(kpts.size());
    for (int i = 0; i < static_cast<int>(kpts.size()); ++i) {
        kpt_grid_sorted.push_back({kpts[i].pt.x, i});
    }
    // 按u坐标升序排列，用于O(log n)二分查找
    std::sort(kpt_grid_sorted.begin(), kpt_grid_sorted.end(),
        [](const KeyPointGridEntry& a, const KeyPointGridEntry& b) {
            return a.u < b.u;
        });
}

} // namespace frontend
} // namespace sfc_core
