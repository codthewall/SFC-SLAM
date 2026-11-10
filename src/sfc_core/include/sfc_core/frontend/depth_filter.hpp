// ================================================================
// 深度采样有效性判定（含深度边缘/飞点过滤）
// ================================================================
//
// 为什么需要它：
//   建 3D 点时只检查了 NaN 和 [min_depth, max_depth] 两个条件，没有检查
//   "这个像素是不是落在深度不连续处"。深度图在物体边界（桌面 / 显示器边缘 /
//   前景人物遮住背景）会出现前景与背景混叠的飞点，取到的深度既不是前景的
//   也不是背景的，建出来的 3D 点会落在两者之间。
//   这种点一旦进了地图，就成了每帧都参与位姿优化的"假锚点"：重投影残差很小，
//   但它把位姿往错误方向拉，是纯 VO 姿态（尤其 roll）缓慢漂移的主因之一。
//   freiburg1_desk2 这种"桌面 + 远处墙面"的场景尤其明显。
//
//   config/tracking.yaml 里早就有 depth_edge_thresh，但从 config_loader 读进来
//   之后没有任何消费者 —— 本模块就是它的消费者。
//
// 判据（全部满足才可用）：
//   1. 深度图是 CV_32FC1 且非空；
//   2. 中心深度有限、> 0；
//   3. 中心深度落在 (min_depth, max_depth]；
//   4. 上下左右 r 像素内的邻域深度与中心深度之差不超过 rel_edge_thresh * depth。
//      rel_edge_thresh <= 0 时跳过第 4 条（等价于关闭过滤）。
// ================================================================
#ifndef SFC_CORE_FRONTEND_DEPTH_FILTER_HPP
#define SFC_CORE_FRONTEND_DEPTH_FILTER_HPP

#include <opencv2/core.hpp>

namespace sfc_core {
namespace frontend {

/// 邻域取样半径（像素）。±2 px 足以跨过 RGB 与深度图之间 1~2 px 的配准误差，
/// 又不会大到把正常的深度倾斜判成边缘。
constexpr int kDepthEdgeNeighborRadius = 2;

/// 判断一次深度采样能否用来建 3D 点。
/// \param depth_m        已换算成米的 CV_32FC1 深度图（与畸变 RGB 对齐）
/// \param u,v            采样像素（畸变坐标）
/// \param depth          该像素的深度值（米），由调用方读好传入，避免重复取值
/// \param min_depth      有效深度下限（米，开区间）
/// \param max_depth      有效深度上限（米，闭区间）
/// \param rel_edge_thresh 邻域相对差阈值：|depth - 邻域depth| > rel_edge_thresh * depth
///                        即判为边缘点（<= 0 表示关闭该检查）
/// \return true 表示该采样可用
bool isValidDepthSample(const cv::Mat& depth_m, int u, int v, double depth,
                        double min_depth, double max_depth, double rel_edge_thresh);

}  // namespace frontend
}  // namespace sfc_core

#endif  // SFC_CORE_FRONTEND_DEPTH_FILTER_HPP
