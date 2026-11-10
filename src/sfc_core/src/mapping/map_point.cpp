#ifdef sfc_core
#undef sfc_core
#endif

#include "sfc_core/mapping/map_point.hpp"

namespace sfc_core {
namespace mapping {

MapPoint::MapPoint(int id, const Eigen::Vector3d& pos) : id_(id), pos_(pos) {}

void MapPoint::setDescriptor(const cv::Mat& desc) {
    if (!desc.empty()) descriptor_ = desc.clone();
}

void MapPoint::addObservation(int keyframe_id, int keypoint_idx) {
    observations_[keyframe_id] = keypoint_idx;
}

void MapPoint::eraseObservation(int keyframe_id) {
    observations_.erase(keyframe_id);
}

bool MapPoint::hasObservation(int keyframe_id) const {
    return observations_.find(keyframe_id) != observations_.end();
}

} // namespace mapping
} // namespace sfc_core
