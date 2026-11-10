#include "sfc_core/frontend/frame.hpp"

namespace sfc_core {
namespace frontend {

Pose::Pose() : q(1, 0, 0, 0), t(0, 0, 0) {}

Pose::Pose(const Eigen::Quaterniond& q_, const Eigen::Vector3d& t_) : q(q_), t(t_) {}

Eigen::Matrix4d Pose::toMatrix() const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = q.toRotationMatrix();
    T.block<3, 1>(0, 3) = t;
    return T;
}

Frame::Ptr Frame::createFrame(const cv::Mat& rgb_img,
                               const cv::Mat& depth_img,
                               const cv::Mat& ir_img,
                               double timestamp) {
    auto frame = std::make_shared<Frame>();
    frame->timestamp = timestamp;
    frame->rgb_img = rgb_img.clone();
    frame->depth_img = depth_img.clone();
    frame->ir_img = ir_img.clone();
    frame->is_keyframe = false;
    frame->id = 0;

    if (rgb_img.channels() == 3) {
        cv::cvtColor(rgb_img, frame->gray_img, cv::COLOR_BGR2GRAY);
    } else {
        frame->gray_img = rgb_img.clone();
    }

    // Pre-allocate feature containers
    constexpr size_t kExpectedKeypoints = 500;
    frame->pts_3d.reserve(kExpectedKeypoints);
    frame->pts_3d_world.reserve(kExpectedKeypoints);
    frame->depth_valid.reserve(kExpectedKeypoints);

    return frame;
}

} // namespace frontend
} // namespace sfc_core
