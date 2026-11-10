// 必须在最开头清除 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_MAPPING_MAP_HPP_
#define SFC_CORE_MAPPING_MAP_HPP_

#include <memory>
#include <vector>
#include <unordered_map>
#include <Eigen/Dense>

namespace sfc_core {
namespace mapping {

class MapPoint;
class KeyFrame;

class Map {
public:
    using Ptr = std::shared_ptr<Map>;
    Map() = default;

    void addMapPoint(MapPoint* pMP);
    MapPoint* getMapPoint(int id) const;
    std::vector<MapPoint*> getAllMapPoints() const;
    void setReferenceMapPoints(const std::vector<MapPoint*>& vpPoints);

    void addKeyFrame(KeyFrame* pKF);
    KeyFrame* getKeyFrame(int frame_id) const;
    std::vector<KeyFrame*> getAllKeyFrames() const;

private:
    std::unordered_map<int, MapPoint*> all_map_points_by_id_;
    std::vector<MapPoint*> all_keyed_map_points_;
    std::unordered_map<int, KeyFrame*> all_key_frames_by_id_;
    std::vector<KeyFrame*> all_key_frames_;
    std::vector<MapPoint*> reference_map_points_;
};

} // namespace mapping
} // namespace sfc_core

#endif // SFC_CORE_MAPPING_MAP_HPP_
