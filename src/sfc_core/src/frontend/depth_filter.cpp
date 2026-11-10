// ================================================================
// 深度采样有效性判定实现
// ================================================================
// 判据与参数语义见 include/sfc_core/frontend/depth_filter.hpp。
// 纯计算，无 Eigen / ROS 依赖，便于单测（test/test_depth_filter.cpp）。
// ================================================================
#include "sfc_core/frontend/depth_filter.hpp"

#include <cmath>

namespace sfc_core {
namespace frontend {

namespace {

/// 读一个邻域像素的深度；越界或无效（NaN / 非正）返回 false。
bool readNeighborDepth(const cv::Mat& depth_m, int u, int v, float* out_depth) {
    if (u < 0 || v < 0 || u >= depth_m.cols || v >= depth_m.rows) return false;
    const float d = depth_m.at<float>(v, u);
    if (!std::isfinite(d) || d <= 0.0f) return false;
    *out_depth = d;
    return true;
}

}  // namespace

bool isValidDepthSample(const cv::Mat& depth_m, int u, int v, double depth,
                        double min_depth, double max_depth, double rel_edge_thresh) {
    if (depth_m.empty() || depth_m.type() != CV_32FC1) return false;
    if (!std::isfinite(depth) || depth <= 0.0) return false;
    if (depth <= min_depth || depth > max_depth) return false;
    if (rel_edge_thresh <= 0.0) return true;

    const int r = kDepthEdgeNeighborRadius;
    const int du[4] = {r, -r, 0, 0};
    const int dv[4] = {0, 0, r, -r};
    const double tolerance = rel_edge_thresh * depth;
    for (int k = 0; k < 4; ++k) {
        float neighbor = 0.0f;
        if (!readNeighborDepth(depth_m, u + du[k], v + dv[k], &neighbor)) continue;
        if (std::fabs(static_cast<double>(neighbor) - depth) > tolerance) return false;
    }
    return true;
}

}  // namespace frontend
}  // namespace sfc_core
