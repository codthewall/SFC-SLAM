// 必须在最开头清除 g2o CMake 注入的 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_FRONTEND_FRAME_HPP_
#define SFC_CORE_FRONTEND_FRAME_HPP_

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <Eigen/Dense>

namespace sfc_core {
namespace frontend {

// ================================================================
// Pose — 相机位姿（4x4 SE3，定义在最外层，避免跨命名空间引用冲突）
// ================================================================
struct Pose {
    Eigen::Quaterniond q;
    Eigen::Vector3d t;

    Pose();
    Pose(const Eigen::Quaterniond& q_, const Eigen::Vector3d& t_);
    Eigen::Matrix4d toMatrix() const;
};

class Frame {
public:
    using Ptr = std::shared_ptr<Frame>;

    static Ptr createFrame(const cv::Mat& rgb_img,
                          const cv::Mat& depth_img,
                          const cv::Mat& ir_img,
                          double timestamp);

    double timestamp = 0.0;
    cv::Mat rgb_img;
    cv::Mat gray_img;
    cv::Mat depth_img;
    cv::Mat ir_img;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> undistorted_keypoints;
    cv::Mat descriptors;
    std::vector<Eigen::Vector3d> pts_3d;       // camera 坐标系 3D 点
    std::vector<Eigen::Vector3d> pts_3d_world; // 世界坐标系 3D 点（用于投影匹配）
    std::vector<bool> depth_valid;
    Pose pose;
    bool is_keyframe = false;
    unsigned long id = 0;
};

} // namespace frontend
} // namespace sfc_core

#endif // SFC_CORE_FRONTEND_FRAME_HPP_
