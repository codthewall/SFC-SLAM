// 必须在最开头清除 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_MAPPING_KEY_FRAME_HPP_
#define SFC_CORE_MAPPING_KEY_FRAME_HPP_

#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "sfc_core/frontend/frame.hpp"

namespace sfc_core {
namespace mapping {

class MapPoint;

class KeyFrame {
public:
    using Ptr = std::shared_ptr<KeyFrame>;

    explicit KeyFrame(int frame_id);

    void setPose(const struct frontend::Pose& pose);
    const struct frontend::Pose& getPose() const { return pose_; }

    void setKeys(const std::vector<cv::KeyPoint>& keys);
    void setUndistortedKeys(const std::vector<cv::KeyPoint>& keys);
    void setDescriptors(const cv::Mat& desc);
    void setStereoPoints(const std::vector<Eigen::Vector3d>& pts);
    void setDepthValid(const std::vector<bool>& valid);

    void setCameraParams(double fx, double fy, double cx, double cy,
                         double invfx, double invfy, double b, double th_depth);

    void addMapPoint(int keypoint_idx, MapPoint* pMP);
    void eraseMapPoint(int keypoint_idx);
    MapPoint* getMapPoint(int keypoint_idx) const;
    const std::map<int, MapPoint*>& getMapPoints() const { return map_points_; }
    std::vector<MapPoint*> getMapPointMatches() const;

    int frameId() const { return frame_id_; }

    const std::vector<cv::KeyPoint>& getKeys() const { return keys_; }
    const std::vector<cv::KeyPoint>& getUndistortedKeys() const { return undist_keys_; }
    const cv::Mat& getDescriptors() const { return descriptors_; }
    const std::vector<Eigen::Vector3d>& getStereoPoints() const { return stereo_points_; }
    const std::vector<bool>& getDepthValid() const { return depth_valid_; }

private:
    int frame_id_ = -1;
    struct frontend::Pose pose_;
    std::vector<cv::KeyPoint> keys_;
    std::vector<cv::KeyPoint> undist_keys_;
    cv::Mat descriptors_;
    std::vector<Eigen::Vector3d> stereo_points_;
    std::vector<bool> depth_valid_;
    double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0;
    double invfx_ = 0, invfy_ = 0, b_ = 0, th_depth_ = 0;
    std::map<int, MapPoint*> map_points_;
};

} // namespace mapping
} // namespace sfc_core

#endif // SFC_CORE_MAPPING_KEY_FRAME_HPP_
