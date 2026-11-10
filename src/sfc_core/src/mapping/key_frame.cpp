#ifdef sfc_core
#undef sfc_core
#endif

#include "sfc_core/mapping/key_frame.hpp"
#include "sfc_core/mapping/map_point.hpp"
#include "sfc_core/frontend/frame.hpp"

namespace sfc_core {
namespace mapping {

KeyFrame::KeyFrame(int frame_id) : frame_id_(frame_id) {}

void KeyFrame::setPose(const frontend::Pose& pose) { pose_ = pose; }

void KeyFrame::setKeys(const std::vector<cv::KeyPoint>& keys) { keys_ = keys; }

void KeyFrame::setUndistortedKeys(const std::vector<cv::KeyPoint>& keys) { undist_keys_ = keys; }

void KeyFrame::setDescriptors(const cv::Mat& desc) {
    if (!desc.empty()) descriptors_ = desc.clone();
}

void KeyFrame::setStereoPoints(const std::vector<Eigen::Vector3d>& pts) { stereo_points_ = pts; }

void KeyFrame::setDepthValid(const std::vector<bool>& valid) { depth_valid_ = valid; }

void KeyFrame::setCameraParams(double fx, double fy, double cx, double cy,
                              double invfx, double invfy, double b, double th_depth) {
    fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
    invfx_ = invfx; invfy_ = invfy; b_ = b; th_depth_ = th_depth;
}

void KeyFrame::addMapPoint(int keypoint_idx, MapPoint* pMP) {
    if (pMP) {
        map_points_[keypoint_idx] = pMP;
    }
}

void KeyFrame::eraseMapPoint(int keypoint_idx) {
    map_points_.erase(keypoint_idx);
}

MapPoint* KeyFrame::getMapPoint(int keypoint_idx) const {
    auto it = map_points_.find(keypoint_idx);
    return it != map_points_.end() ? it->second : nullptr;
}

std::vector<MapPoint*> KeyFrame::getMapPointMatches() const {
    std::vector<MapPoint*> matches;
    matches.reserve(map_points_.size());
    for (const auto& kv : map_points_) {
        if (kv.second) {
            matches.push_back(kv.second);
        }
    }
    return matches;
}

} // namespace mapping
} // namespace sfc_core
