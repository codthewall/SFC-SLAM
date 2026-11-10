// 必须在最开头清除 sfc_core 宏
#ifdef sfc_core
#undef sfc_core
#endif

#ifndef SFC_CORE_UTILS_CAMERA_UTILS_HPP_
#define SFC_CORE_UTILS_CAMERA_UTILS_HPP_

#include <opencv2/opencv.hpp>

namespace sfc_core {
namespace utils {

// 投影模型
inline cv::Point2f project(const cv::Point3f& p, double fx, double fy, double cx, double cy) {
    if (p.z <= 0) return cv::Point2f(-1, -1);
    double invz = 1.0 / p.z;
    return cv::Point2f(static_cast<float>(fx * p.x * invz + cx),
                       static_cast<float>(fy * p.y * invz + cy));
}

} // namespace utils
} // namespace sfc_core

#endif // SFC_CORE_UTILS_CAMERA_UTILS_HPP_
