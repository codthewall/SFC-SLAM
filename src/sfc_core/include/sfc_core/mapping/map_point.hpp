// 必须在最开头清除 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_MAPPING_MAP_POINT_HPP_
#define SFC_CORE_MAPPING_MAP_POINT_HPP_

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace sfc_core {
namespace mapping {

class MapPoint {
public:
    explicit MapPoint(int id, const Eigen::Vector3d& pos);
    MapPoint() = default;

    int id() const { return id_; }
    const Eigen::Vector3d& getWorldPos() const { return pos_; }
    void setWorldPos(const Eigen::Vector3d& pos) { pos_ = pos; }

    void setDescriptor(const cv::Mat& desc);
    const cv::Mat& getDescriptor() const { return descriptor_; }

    void addObservation(int keyframe_id, int keypoint_idx);
    void eraseObservation(int keyframe_id);
    bool hasObservation(int keyframe_id) const;
    int observations() const { return static_cast<int>(observations_.size()); }
    const std::map<int, int>& getObservations() const { return observations_; }

    void setReferenceKeyFrame(int keyframe_id) { ref_kf_id_ = keyframe_id; }
    int referenceKeyFrame() const { return ref_kf_id_; }
    void setFirstKFid(int keyframe_id) { first_kf_id_ = keyframe_id; }
    int firstKFid() const { return first_kf_id_; }

    void setBad(bool bad = true) { is_bad_ = bad; }
    bool isBad() const { return is_bad_; }

    void increaseVisible(int n = 1) { visible_count_ += n; }
    void increaseFound(int n = 1) { found_count_ += n; }
    void setFound(int n) { found_count_ = n; }
    int visibleCount() const { return visible_count_; }
    int foundCount() const { return found_count_; }
    double foundRatio() const { return visible_count_ > 0 ? static_cast<double>(found_count_) / visible_count_ : 0.0; }

    void setNormal(const Eigen::Vector3d& normal) { normal_ = normal; }
    const Eigen::Vector3d& normal() const { return normal_; }

private:
    int id_ = -1;
    Eigen::Vector3d pos_ = Eigen::Vector3d::Zero();
    cv::Mat descriptor_;
    std::map<int, int> observations_; // keyframe_id -> keypoint_idx
    int ref_kf_id_ = -1;
    int first_kf_id_ = -1;
    bool is_bad_ = false;
    int visible_count_ = 1;
    int found_count_ = 1;
    Eigen::Vector3d normal_ = Eigen::Vector3d::Zero();
};

} // namespace mapping
} // namespace sfc_core

#endif // SFC_CORE_MAPPING_MAP_POINT_HPP_
