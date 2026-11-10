// ================================================================
// 深度过滤单测（自带 main，不依赖 gtest）
// ================================================================
// 覆盖 isValidDepthSample() 的每一条判据：类型、空图、NaN、非正、
// 上下限、深度边缘、邻域越界、关闭边缘检查。
// ================================================================
#include "sfc_core/frontend/depth_filter.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool ok, const char* name) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("  FAIL  %s\n", name);
    } else {
        std::printf("  ok    %s\n", name);
    }
}

constexpr double kMinDepth = 0.1;
constexpr double kMaxDepth = 8.0;
constexpr double kEdge = 0.2;

cv::Mat makeFlat(float value, int rows = 7, int cols = 7) {
    return cv::Mat(rows, cols, CV_32FC1, cv::Scalar(value));
}

}  // namespace

int main() {
    std::printf("==== test_depth_filter ====\n");

    // [1] 平坦区域、深度合法 → 可用
    {
        const cv::Mat flat = makeFlat(2.0f);
        std::printf("[1] 平坦区域\n");
        check(sfc_core::frontend::isValidDepthSample(flat, 3, 3, 2.0, kMinDepth, kMaxDepth, kEdge),
              "flat_valid");
    }

    // [2] 深度 0 / 负值 / NaN → 不可用
    {
        const cv::Mat flat = makeFlat(2.0f);
        std::printf("[2] 非法深度值\n");
        check(!sfc_core::frontend::isValidDepthSample(flat, 3, 3, 0.0, kMinDepth, kMaxDepth, kEdge),
              "zero_invalid");
        check(!sfc_core::frontend::isValidDepthSample(flat, 3, 3, -1.0, kMinDepth, kMaxDepth, kEdge),
              "negative_invalid");
        check(!sfc_core::frontend::isValidDepthSample(
                  flat, 3, 3, std::numeric_limits<double>::quiet_NaN(), kMinDepth, kMaxDepth, kEdge),
              "nan_invalid");
    }

    // [3] 超出 [min_depth, max_depth] → 不可用
    {
        const cv::Mat flat = makeFlat(9.0f);
        std::printf("[3] 深度超范围\n");
        check(!sfc_core::frontend::isValidDepthSample(flat, 3, 3, 9.0, kMinDepth, kMaxDepth, kEdge),
              "too_far_invalid");
        check(!sfc_core::frontend::isValidDepthSample(flat, 3, 3, 0.05, kMinDepth, kMaxDepth, kEdge),
              "too_near_invalid");
    }

    // [4] 深度阶跃：中心在 2 m，右侧 2 px 是 6 m → 判为边缘点
    {
        cv::Mat step = makeFlat(2.0f);
        for (int v = 0; v < step.rows; ++v) {
            for (int u = 4; u < step.cols; ++u) step.at<float>(v, u) = 6.0f;
        }
        std::printf("[4] 深度边缘\n");
        check(!sfc_core::frontend::isValidDepthSample(step, 3, 3, 2.0, kMinDepth, kMaxDepth, kEdge),
              "edge_rejected");
        // 远离边缘的同一个点 → 可用
        check(sfc_core::frontend::isValidDepthSample(step, 1, 3, 2.0, kMinDepth, kMaxDepth, kEdge),
              "away_from_edge_valid");
        // 关闭边缘检查（thresh=0）→ 边缘点也放过
        check(sfc_core::frontend::isValidDepthSample(step, 3, 3, 2.0, kMinDepth, kMaxDepth, 0.0),
              "edge_check_disabled");
    }

    // [5] 邻域有无效像素（0）→ 跳过该方向，仍可用
    {
        cv::Mat hole = makeFlat(2.0f);
        hole.at<float>(3, 5) = 0.0f;
        std::printf("[5] 邻域空洞\n");
        check(sfc_core::frontend::isValidDepthSample(hole, 3, 3, 2.0, kMinDepth, kMaxDepth, kEdge),
              "neighbor_hole_skipped");
    }

    // [6] 图像边界（邻域越界）→ 跳过越界方向，仍可用
    {
        const cv::Mat flat = makeFlat(2.0f);
        std::printf("[6] 图像边界\n");
        check(sfc_core::frontend::isValidDepthSample(flat, 0, 0, 2.0, kMinDepth, kMaxDepth, kEdge),
              "corner_valid");
    }

    // [7] 非 CV_32FC1 或空图 → 不可用
    {
        const cv::Mat u16(7, 7, CV_16UC1, cv::Scalar(2000));
        const cv::Mat empty;
        std::printf("[7] 类型与空图\n");
        check(!sfc_core::frontend::isValidDepthSample(u16, 3, 3, 2.0, kMinDepth, kMaxDepth, kEdge),
              "wrong_type_invalid");
        check(!sfc_core::frontend::isValidDepthSample(empty, 0, 0, 2.0, kMinDepth, kMaxDepth, kEdge),
              "empty_invalid");
    }

    std::printf("==== %d checks, %d failures ====\n", g_checks, g_failures);
    std::printf("%s\n", g_failures == 0 ? "RESULT: PASSED" : "RESULT: FAILED");
    return g_failures == 0 ? 0 : 1;
}
