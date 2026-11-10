#ifdef sfc_core
#undef sfc_core
#endif

#include "sfc_core/mapping/map.hpp"
#include "sfc_core/mapping/map_point.hpp"
#include "sfc_core/mapping/key_frame.hpp"

namespace sfc_core {
namespace mapping {

void Map::addMapPoint(MapPoint* pMP) {
    if (!pMP) return;

    const int id = pMP->id();
    // 如果已存在，不替换（前端只创建新 ID 的点，后端不会传重复 ID 上来）
    if (all_map_points_by_id_.count(id)) return;
    all_map_points_by_id_.emplace(id, pMP);
    all_keyed_map_points_.push_back(pMP);
}

MapPoint* Map::getMapPoint(int id) const {
    auto it = all_map_points_by_id_.find(id);
    return it != all_map_points_by_id_.end() ? it->second : nullptr;
}

std::vector<MapPoint*> Map::getAllMapPoints() const {
    return all_keyed_map_points_;
}

void Map::setReferenceMapPoints(const std::vector<MapPoint*>& vpPoints) {
    reference_map_points_ = vpPoints;
}

void Map::addKeyFrame(KeyFrame* pKF) {
    if (!pKF) return;

    const int frame_id = pKF->frameId();
    if (all_key_frames_by_id_.count(frame_id)) return;
    all_key_frames_by_id_.emplace(frame_id, pKF);
    all_key_frames_.push_back(pKF);
}

KeyFrame* Map::getKeyFrame(int frame_id) const {
    auto it = all_key_frames_by_id_.find(frame_id);
    return it != all_key_frames_by_id_.end() ? it->second : nullptr;
}

std::vector<KeyFrame*> Map::getAllKeyFrames() const {
    return all_key_frames_;
}

} // namespace mapping
} // namespace sfc_core
