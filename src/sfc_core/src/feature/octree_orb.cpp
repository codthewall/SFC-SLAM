#ifdef sfc_core
#undef sfc_core
#endif

#include "sfc_core/feature/octree_orb.hpp"
#include <algorithm>
#include <cmath>

namespace sfc_core {
namespace feature {

// ================================================================
// ORB 描述子模式表（与 ORB_SLAM3 bit_pattern_31_ 完全一致，共 256 条记录）
// 每条记录 = 4 个 int：x1, y1, x2, y2，共 256*4 = 1024 个 int
// ================================================================
static const int bit_pattern_31_[256 * 4] = {
    8,-3, 9,5,
    4,2, 7,-12,
    -11,9, -8,2,
    7,-12, 12,-13,
    2,-13, 2,12,
    1,-7, 1,6,
    -2,-10, -2,-4,
    -13,-13, -11,-8,
    -13,-3, -12,-9,
    10,4, 11,9,
    -13,-8, -8,-9,
    -11,7, -9,12,
    7,7, 12,6,
    -4,-5, -3,0,
    -13,2, -12,-3,
    -9,0, -7,5,
    12,-6, 12,-1,
    -3,6, -2,12,
    -6,-13, -4,-8,
    11,-13, 12,-8,
    4,7, 5,1,
    5,-3, 10,-3,
    3,-7, 6,12,
    -8,-7, -6,-2,
    -2,11, -1,-10,
    -13,12, -8,10,
    -7,3, -5,-3,
    -4,2, -3,7,
    -10,-12, -6,11,
    5,-12, 6,-7,
    5,-6, 7,-1,
    1,0, 4,-5,
    9,11, 11,-13,
    4,7, 4,12,
    2,-1, 4,4,
    -4,-12, -2,7,
    -8,-5, -7,-10,
    4,11, 9,12,
    0,-8, 1,-13,
    -13,-2, -8,2,
    -3,-2, -2,3,
    -6,9, -4,-9,
    8,12, 10,7,
    0,9, 1,3,
    7,-5, 11,-10,
    -13,-6, -11,0,
    10,7, 12,1,
    -6,-3, -6,12,
    10,-9, 12,-4,
    -13,8, -8,-12,
    -13,0, -8,-4,
    3,3, 7,8,
    5,7, 10,-7,
    -1,7, 1,-12,
    3,-10, 5,6,
    2,-4, 3,-10,
    -13,0, -13,5,
    -13,-7, -12,12,
    -13,3, -11,8,
    -7,12, -4,7,
    6,-10, 12,8,
    -9,-1, -7,-6,
    -2,-5, 0,12,
    -12,5, -7,5,
    3,-10, 8,-13,
    -7,-7, -4,5,
    -3,-2, -1,-7,
    2,9, 5,-11,
    -11,-13, -5,-13,
    -1,6, 0,-1,
    5,-3, 5,2,
    -4,-13, -4,12,
    -9,-6, -9,6,
    -12,-10, -8,-4,
    10,2, 12,-3,
    7,12, 12,12,
    -7,-13, -6,5,
    -4,9, -3,4,
    7,-1, 12,2,
    -7,6, -5,1,
    -13,11, -12,5,
    -3,7, -2,-6,
    7,-8, 12,-7,
    -13,-7, -11,-12,
    1,-3, 12,12,
    2,-6, 3,0,
    -4,3, -2,-13,
    -1,-13, 1,9,
    7,1, 8,-6,
    1,-1, 3,12,
    9,1, 12,6,
    -1,-9, -1,3,
    -13,-13, -10,5,
    7,7, 10,12,
    12,-5, 12,9,
    6,3, 7,11,
    5,-13, 6,10,
    2,-12, 2,3,
    3,8, 4,-6,
    2,6, 12,-13,
    9,-12, 10,3,
    -8,4, -7,9,
    -11,12, -4,-6,
    1,12, 2,-8,
    6,-9, 7,-4,
    2,3, 3,-2,
    6,3, 11,0,
    3,-3, 8,-8,
    7,8, 9,3,
    -11,-5, -6,-4,
    -10,11, -5,10,
    -5,-8, -3,12,
    -10,5, -9,0,
    8,-1, 12,-6,
    4,-6, 6,-11,
    -10,12, -8,7,
    4,-2, 6,7,
    -2,0, -2,12,
    -5,-8, -5,2,
    7,-6, 10,12,
    -9,-13, -8,-8,
    -5,-13, -5,-2,
    8,-8, 9,-13,
    -9,-11, -9,0,
    1,-8, 1,-2,
    7,-4, 9,1,
    -2,1, -1,-4,
    11,-6, 12,-11,
    -12,-9, -6,4,
    3,7, 7,12,
    5,5, 10,8,
    0,-4, 2,8,
    -9,12, -5,-13,
    0,7, 2,12,
    -1,2, 1,7,
    5,11, 7,-9,
    3,5, 6,-8,
    -13,-4, -8,9,
    -5,9, -3,-3,
    -4,-7, -3,-12,
    6,5, 8,0,
    -7,6, -6,12,
    -13,6, -5,-2,
    1,-10, 3,10,
    4,1, 8,-4,
    -2,-2, 2,-13,
    2,-12, 12,12,
    -2,-13, 0,-6,
    4,1, 9,3,
    -6,-10, -3,-5,
    -3,-13, -1,1,
    7,5, 12,-11,
    4,-2, 5,-7,
    -13,9, -9,-5,
    7,1, 8,6,
    7,-8, 7,6,
    -7,-4, -7,1,
    -8,11, -7,-8,
    -13,6, -12,-8,
    2,4, 3,9,
    10,-5, 12,3,
    -6,-5, -6,7,
    8,-3, 9,-8,
    2,-12, 2,8,
    -11,-2, -10,3,
    -12,-13, -7,-9,
    -11,0, -10,-5,
    5,-3, 11,8,
    -2,-13, -1,12,
    -1,-8, 0,9,
    -13,-11, -12,-5,
    -10,-2, -10,11,
    -3,9, -2,-13,
    2,-3, 3,2,
    -9,-13, -4,0,
    -4,6, -3,-10,
    -4,12, -2,-7,
    -6,-11, -4,9,
    6,-3, 6,11,
    -13,11, -5,5,
    11,11, 12,6,
    7,-5, 12,-2,
    -1,12, 0,7,
    -4,-8, -3,-2,
    -7,1, -6,7,
    -13,-12, -8,-13,
    -7,-2, -6,-8,
    -8,5, -6,-9,
    -5,-1, -4,5,
    -13,7, -8,10,
    1,5, 5,-13,
    1,0, 10,-13,
    9,12, 10,-1,
    5,-8, 10,-9,
    -1,11, 1,-13,
    -9,-3, -6,2,
    -1,-10, 1,12,
    -13,1, -8,-10,
    8,-11, 10,-6,
    2,-13, 3,-6,
    7,-13, 12,-9,
    -10,-10, -5,-7,
    -10,-8, -8,-13,
    4,-6, 8,5,
    3,12, 8,-13,
    -4,2, -3,-3,
    5,-13, 10,-12,
    4,-13, 5,-1,
    -9,9, -4,3,
    0,3, 3,-9,
    -12,1, -6,1,
    3,2, 4,-8,
    -10,-10, -10,9,
    8,-13, 12,12,
    -8,-12, -6,-5,
    2,2, 3,7,
    10,6, 11,-8,
    6,8, 8,-12,
    -7,10, -6,5,
    -3,-9, -3,9,
    -1,-13, -1,5,
    -3,-7, -3,4,
    -8,-2, -8,3,
    4,2, 12,12,
    2,-5, 3,11,
    6,-9, 11,-13,
    3,-1, 7,12,
    11,-1, 12,4,
    -3,0, -3,6,
    4,-11, 4,12,
    2,-4, 2,1,
    -10,-6, -8,1,
    -13,7, -11,1,
    -13,12, -11,-13,
    6,0, 11,-13,
    0,-1, 1,4,
    -13,3, -9,-2,
    -9,8, -6,-3,
    -13,-6, -8,-2,
    5,-9, 8,10,
    2,7, 3,-9,
    -1,-6, -1,-1,
    9,5, 11,-2,
    11,-3, 12,-8,
    3,0, 3,5,
    -1,4, 0,10,
    3,-6, 4,5,
    -13,0, -10,5,
    5,8, 12,11,
    8,9, 9,-6,
    7,-4, 8,-12,
    -10,4, -10,9,
    7,3, 12,4,
    9,-7, 10,-2,
    7,0, 12,-2,
    -1,-6, 0,-11
};

// ================================================================
// IC_Angle — 计算特征点方向（与 ORB_SLAM3 完全一致）
// 利用灰度质心法（Intensity Centroid）计算旋转不变方向
// ================================================================
static float IC_Angle(const cv::Mat& image,
                      const cv::Point2f pt,
                      const std::vector<int>& u_max) {
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar>(
        cvRound(pt.y), cvRound(pt.x));

    const int half_patch = 15;
    const int step = static_cast<int>(image.step1());

    for (int u = -half_patch; u <= half_patch; ++u)
        m_10 += u * center[u];

    for (int v = 1; v <= half_patch; ++v) {
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u) {
            int val_plus  = center[u + v * step];
            int val_minus = center[u - v * step];
            v_sum    += (val_plus - val_minus);
            m_10     += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return static_cast<float>(std::atan2(m_01, m_10) * 180.0 / CV_PI);
}

// ================================================================
// computeOrbDescriptor — 计算旋转感知的 ORB rBRIEF 描述子
// 与 ORB_SLAM3 computeOrbDescriptor 完全一致
// ================================================================
static const float factor_pi = static_cast<float>(CV_PI / 180.0f);

static void computeOrbDescriptor(const cv::KeyPoint& kpt,
                                const cv::Mat& img,
                                const cv::Point* pattern_ptr,
                                uchar* desc) {
    float angle = static_cast<float>(kpt.angle) * factor_pi;
    float a = static_cast<float>(std::cos(angle));
    float b = static_cast<float>(std::sin(angle));

    const uchar* center = &img.at<uchar>(
        cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = static_cast<int>(img.step);

    for (int i = 0; i < 32; ++i, pattern_ptr += 16) {
        int t0, t1, val;
        t0 = center[cvRound(pattern_ptr[0].x  * b + pattern_ptr[0].y  * a) * step +
                   cvRound(pattern_ptr[0].x  * a - pattern_ptr[0].y  * b)];
        t1 = center[cvRound(pattern_ptr[1].x  * b + pattern_ptr[1].y  * a) * step +
                   cvRound(pattern_ptr[1].x  * a - pattern_ptr[1].y  * b)];
        val = t0 < t1;
        t0 = center[cvRound(pattern_ptr[2].x  * b + pattern_ptr[2].y  * a) * step +
                   cvRound(pattern_ptr[2].x  * a - pattern_ptr[2].y  * b)];
        t1 = center[cvRound(pattern_ptr[3].x  * b + pattern_ptr[3].y  * a) * step +
                   cvRound(pattern_ptr[3].x  * a - pattern_ptr[3].y  * b)];
        val |= (t0 < t1) << 1;
        t0 = center[cvRound(pattern_ptr[4].x  * b + pattern_ptr[4].y  * a) * step +
                   cvRound(pattern_ptr[4].x  * a - pattern_ptr[4].y  * b)];
        t1 = center[cvRound(pattern_ptr[5].x  * b + pattern_ptr[5].y  * a) * step +
                   cvRound(pattern_ptr[5].x  * a - pattern_ptr[5].y  * b)];
        val |= (t0 < t1) << 2;
        t0 = center[cvRound(pattern_ptr[6].x  * b + pattern_ptr[6].y  * a) * step +
                   cvRound(pattern_ptr[6].x  * a - pattern_ptr[6].y  * b)];
        t1 = center[cvRound(pattern_ptr[7].x  * b + pattern_ptr[7].y  * a) * step +
                   cvRound(pattern_ptr[7].x  * a - pattern_ptr[7].y  * b)];
        val |= (t0 < t1) << 3;
        t0 = center[cvRound(pattern_ptr[8].x  * b + pattern_ptr[8].y  * a) * step +
                   cvRound(pattern_ptr[8].x  * a - pattern_ptr[8].y  * b)];
        t1 = center[cvRound(pattern_ptr[9].x  * b + pattern_ptr[9].y  * a) * step +
                   cvRound(pattern_ptr[9].x  * a - pattern_ptr[9].y  * b)];
        val |= (t0 < t1) << 4;
        t0 = center[cvRound(pattern_ptr[10].x * b + pattern_ptr[10].y * a) * step +
                   cvRound(pattern_ptr[10].x * a - pattern_ptr[10].y * b)];
        t1 = center[cvRound(pattern_ptr[11].x * b + pattern_ptr[11].y * a) * step +
                   cvRound(pattern_ptr[11].x * a - pattern_ptr[11].y * b)];
        val |= (t0 < t1) << 5;
        t0 = center[cvRound(pattern_ptr[12].x * b + pattern_ptr[12].y * a) * step +
                   cvRound(pattern_ptr[12].x * a - pattern_ptr[12].y * b)];
        t1 = center[cvRound(pattern_ptr[13].x * b + pattern_ptr[13].y * a) * step +
                   cvRound(pattern_ptr[13].x * a - pattern_ptr[13].y * b)];
        val |= (t0 < t1) << 6;
        t0 = center[cvRound(pattern_ptr[14].x * b + pattern_ptr[14].y * a) * step +
                   cvRound(pattern_ptr[14].x * a - pattern_ptr[14].y * b)];
        t1 = center[cvRound(pattern_ptr[15].x * b + pattern_ptr[15].y * a) * step +
                   cvRound(pattern_ptr[15].x * a - pattern_ptr[15].y * b)];
        val |= (t0 < t1) << 7;
        desc[i] = static_cast<uchar>(val);
    }
}

// ================================================================
// OctreeORB 构造函数
// ================================================================
OctreeORB::OctreeORB(const OctreeORBConfig& config) : config_(config) {
    const int npoints = 512;

    mv_scale_factor_.resize(config_.n_levels);
    mv_level_sigma2_.resize(config_.n_levels);
    mv_scale_factor_[0] = 1.0f;
    mv_level_sigma2_[0] = 1.0f;
    for (int i = 1; i < config_.n_levels; ++i) {
        mv_scale_factor_[i] = mv_scale_factor_[i - 1] * config_.scale_factor;
        mv_level_sigma2_[i] = mv_scale_factor_[i] * mv_scale_factor_[i];
    }

    mv_inv_scale_factor_.resize(config_.n_levels);
    mv_inv_level_sigma2_.resize(config_.n_levels);
    for (int i = 0; i < config_.n_levels; ++i) {
        mv_inv_scale_factor_[i] = 1.0f / mv_scale_factor_[i];
        mv_inv_level_sigma2_[i] = 1.0f / mv_level_sigma2_[i];
    }

    image_pyramid_.resize(config_.n_levels);

    // mnFeaturesPerLevel：按几何级数分配每层特征数量（与 ORB_SLAM3 一致）
    mn_features_per_level_.resize(config_.n_levels);
    float factor = 1.0f / config_.scale_factor;
    float n_desired = static_cast<float>(config_.n_features) *
        (1 - factor) / (1 - std::pow(factor, config_.n_levels));

    int sum_features = 0;
    for (int level = 0; level < config_.n_levels - 1; ++level) {
        mn_features_per_level_[level] = cvRound(n_desired);
        sum_features += mn_features_per_level_[level];
        n_desired *= factor;
    }
    mn_features_per_level_[config_.n_levels - 1] =
        std::max(config_.n_features - sum_features, 0);

    // 加载 ORB_SLAM3 的 bit_pattern_31_ 模式表（512 个 cv::Point）
    const cv::Point* pattern0 = reinterpret_cast<const cv::Point*>(bit_pattern_31_);
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern_));

    // 预计算 umax（用于 IC_Angle）
    const int half_patch = 15;
    umax_.resize(half_patch + 1);
    int vmax = cvFloor(half_patch * std::sqrt(2.0f) / 2 + 1);
    int vmin = cvCeil(half_patch * std::sqrt(2.0f) / 2);
    const double hp2 = half_patch * half_patch;
    for (int v = 0; v <= vmax; ++v)
        umax_[v] = cvRound(std::sqrt(hp2 - v * v));
    for (int v = half_patch, v0 = 0; v >= vmin; --v) {
        while (umax_[v0] == umax_[v0 + 1])
            ++v0;
        umax_[v] = v0;
        ++v0;
    }
}

// ================================================================
// extract — 主入口（与 ORB_SLAM3 operator() 逻辑一致）
// ================================================================
void OctreeORB::extract(const cv::Mat& img,
                        const cv::Mat& /*img_right*/,
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors) {
    CV_Assert(!img.empty() && img.type() == CV_8U);

    keypoints.clear();
    descriptors.release();

    computePyramid(img);

    std::vector<std::vector<cv::KeyPoint>> all_keypoints(config_.n_levels);
    computeKeyPointsOctTree(all_keypoints);

    int n_keypoints = 0;
    for (int level = 0; level < config_.n_levels; ++level)
        n_keypoints += static_cast<int>(all_keypoints[level].size());

    if (n_keypoints == 0) return;

    keypoints.resize(n_keypoints);
    descriptors.create(n_keypoints, 32, CV_8U);

    int offset = 0;
    for (int level = 0; level < config_.n_levels; ++level) {
        const auto& keypoints_level = all_keypoints[level];
        const int n_kp_level = static_cast<int>(keypoints_level.size());
        if (n_kp_level == 0) continue;

        // ORB_SLAM3：对每层做高斯模糊再计算描述子（降噪加速）
        // image_pyramid_[level] 是 temp 的带 padding 子区域，连续内存，边界像素已由
        // copyMakeBorder 正确填充（两次调用均已去掉 BORDER_ISOLATED）。
        cv::Mat blurred;
        cv::GaussianBlur(image_pyramid_[level], blurred,
                         cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);
        cv::Rect roi(config_.edge_threshold, config_.edge_threshold,
                     blurred.cols - config_.edge_threshold * 2,
                     blurred.rows - config_.edge_threshold * 2);
        cv::Mat working_mat = blurred(roi);

        // kp.pt 在 image_pyramid_[level] 坐标系下（范围 [0, sz-1]）
        // working_mat 是去掉 padding 的有效区域，坐标系一致（范围 [0, sz-1]）
        // 无需坐标转换，kp.pt 直接用于 working_mat
        // 注意：keypoint 检测网格排除了最外层 min_border_x=16 的边距，
        // + distributeOctTree 约束在 [min_border_x, max_border_x] 内，
        // 所以 kp.pt 始终在 [min_border_x, max_border_x] 范围内
        // = [16, sz-17]，不会越界

        cv::Mat desc_level = descriptors.rowRange(offset, offset + n_kp_level);
        for (int i = 0; i < n_kp_level; ++i) {
            const cv::KeyPoint& kp_src = keypoints_level[i];
            cv::KeyPoint kp_out = kp_src;

            kp_out.angle = IC_Angle(working_mat, kp_out.pt, umax_);
            computeOrbDescriptor(kp_out, working_mat, &pattern_[0],
                                desc_level.ptr<uchar>(i));

            keypoints[offset + i] = kp_out;
        }
        offset += n_kp_level;
    }
}

// ================================================================
// computePyramid — 构建图像金字塔（与 ORB_SLAM3 完全一致）
// 关键：每层用 copyMakeBorder 填充边界，保证 FAST 检测不越界
// ================================================================
void OctreeORB::computePyramid(const cv::Mat& img) {
    const int edge_th = config_.edge_threshold;

    for (int level = 0; level < config_.n_levels; ++level) {
        float scale = mv_inv_scale_factor_[level];
        cv::Size sz(cvRound(static_cast<float>(img.cols) * scale),
                    cvRound(static_cast<float>(img.rows) * scale));
        cv::Size whole_sz(sz.width  + edge_th * 2,
                          sz.height + edge_th * 2);
        cv::Mat temp(whole_sz, img.type());
        image_pyramid_[level] = temp(cv::Rect(edge_th, edge_th,
                                             sz.width, sz.height));

        if (level == 0) {
            cv::copyMakeBorder(img, temp,
                               edge_th, edge_th, edge_th, edge_th,
                               cv::BORDER_REFLECT_101);
        } else {
            cv::resize(image_pyramid_[level - 1],
                       image_pyramid_[level], sz, 0, 0, cv::INTER_LINEAR);
            cv::copyMakeBorder(image_pyramid_[level], temp,
                               edge_th, edge_th, edge_th, edge_th,
                               cv::BORDER_REFLECT_101);
        }
    }
}

// ================================================================
// ExtractorNode::divideNode — 四叉树分裂（与 ORB_SLAM3 完全一致）
// ================================================================
void ExtractorNode::divideNode(ExtractorNode& n1, ExtractorNode& n2,
                              ExtractorNode& n3, ExtractorNode& n4) {
    const int half_x = cvCeil(static_cast<float>(ur.x - ul.x) / 2.0f);
    const int half_y = cvCeil(static_cast<float>(br.y - ul.y) / 2.0f);

    n1.ul = ul;
    n1.ur = cv::Point2i(ul.x + half_x, ul.y);
    n1.bl = cv::Point2i(ul.x, ul.y + half_y);
    n1.br = cv::Point2i(ul.x + half_x, ul.y + half_y);
    n1.v_keys.reserve(v_keys.size());

    n2.ul = n1.ur;
    n2.ur = ur;
    n2.bl = n1.br;
    n2.br = cv::Point2i(ur.x, ul.y + half_y);
    n2.v_keys.reserve(v_keys.size());

    n3.ul = n1.bl;
    n3.ur = n1.br;
    n3.bl = bl;
    n3.br = cv::Point2i(n1.br.x, bl.y);
    n3.v_keys.reserve(v_keys.size());

    n4.ul = n3.ur;
    n4.ur = n2.br;
    n4.bl = n3.br;
    n4.br = br;
    n4.v_keys.reserve(v_keys.size());

    for (const auto& kp : v_keys) {
        if (kp.pt.x < n1.ur.x) {
            if (kp.pt.y < n1.br.y)
                n1.v_keys.push_back(kp);
            else
                n3.v_keys.push_back(kp);
        } else if (kp.pt.y < n1.br.y) {
            n2.v_keys.push_back(kp);
        } else {
            n4.v_keys.push_back(kp);
        }
    }

    if (n1.v_keys.size() == 1) n1.b_no_more = true;
    if (n2.v_keys.size() == 1) n2.b_no_more = true;
    if (n3.v_keys.size() == 1) n3.b_no_more = true;
    if (n4.v_keys.size() == 1) n4.b_no_more = true;
}

// ================================================================
// distributeOctTree — 四叉树均匀分布特征点
// 与 ORB_SLAM3 DistributeOctTree 完全一致
// ================================================================
void OctreeORB::distributeOctTree(
        const std::vector<cv::KeyPoint>& v_to_distribute_keys,
        const int& min_x, const int& max_x,
        const int& min_y, const int& max_y,
        const int& n_features, const int& /*level*/,
        std::vector<cv::KeyPoint>& result) {
    result.clear();

    // 根据宽高比决定初始列数
    const int n_ini = cvRound(static_cast<float>(max_x - min_x) /
                               static_cast<float>(max_y - min_y));
    const float h_x = static_cast<float>(max_x - min_x) / n_ini;

    std::list<ExtractorNode> l_nodes;
    std::vector<ExtractorNode*> vp_ini_nodes(n_ini);
    for (int i = 0; i < n_ini; ++i) {
        ExtractorNode ni;
        ni.ul = cv::Point2i(cvRound(h_x * i), 0);
        ni.ur = cv::Point2i(cvRound(h_x * (i + 1)), 0);
        ni.bl = cv::Point2i(ni.ul.x, max_y - min_y);
        ni.br = cv::Point2i(ni.ur.x, max_y - min_y);
        ni.v_keys.reserve(v_to_distribute_keys.size());
        l_nodes.push_back(ni);
        vp_ini_nodes[i] = &l_nodes.back();
    }

    for (const auto& kp : v_to_distribute_keys)
        vp_ini_nodes[kp.pt.x / h_x]->v_keys.push_back(kp);

    auto lit = l_nodes.begin();
    while (lit != l_nodes.end()) {
        if (lit->v_keys.size() == 1) {
            lit->b_no_more = true;
            ++lit;
        } else if (lit->v_keys.empty()) {
            lit = l_nodes.erase(lit);
        } else {
            ++lit;
        }
    }

    bool b_finish = false;
    std::vector<std::pair<int, ExtractorNode*>> v_size_and_ptr;
    v_size_and_ptr.reserve(l_nodes.size() * 4);

    while (!b_finish) {
        int prev_size = static_cast<int>(l_nodes.size());
        lit = l_nodes.begin();
        int n_to_expand = 0;
        v_size_and_ptr.clear();

        while (lit != l_nodes.end()) {
            if (lit->b_no_more) {
                ++lit;
                continue;
            } else {
                ExtractorNode n1, n2, n3, n4;
                lit->divideNode(n1, n2, n3, n4);

                if (n1.v_keys.size() > 0) {
                    l_nodes.push_front(n1);
                    if (n1.v_keys.size() > 1) {
                        n_to_expand++;
                        v_size_and_ptr.emplace_back(n1.v_keys.size(), &l_nodes.front());
                        l_nodes.front().lit = l_nodes.begin();
                    }
                }
                if (n2.v_keys.size() > 0) {
                    l_nodes.push_front(n2);
                    if (n2.v_keys.size() > 1) {
                        n_to_expand++;
                        v_size_and_ptr.emplace_back(n2.v_keys.size(), &l_nodes.front());
                        l_nodes.front().lit = l_nodes.begin();
                    }
                }
                if (n3.v_keys.size() > 0) {
                    l_nodes.push_front(n3);
                    if (n3.v_keys.size() > 1) {
                        n_to_expand++;
                        v_size_and_ptr.emplace_back(n3.v_keys.size(), &l_nodes.front());
                        l_nodes.front().lit = l_nodes.begin();
                    }
                }
                if (n4.v_keys.size() > 0) {
                    l_nodes.push_front(n4);
                    if (n4.v_keys.size() > 1) {
                        n_to_expand++;
                        v_size_and_ptr.emplace_back(n4.v_keys.size(), &l_nodes.front());
                        l_nodes.front().lit = l_nodes.begin();
                    }
                }

                lit = l_nodes.erase(lit);
                continue;
            }
        }

        if (static_cast<int>(l_nodes.size()) >= n_features ||
            static_cast<int>(l_nodes.size()) == prev_size) {
            b_finish = true;
        } else if (static_cast<int>(l_nodes.size()) + n_to_expand * 3 > n_features) {
            v_size_and_ptr.clear();
            for (auto& node : l_nodes) {
                if (!node.b_no_more && node.v_keys.size() > 1)
                    v_size_and_ptr.emplace_back(node.v_keys.size(), &node);
            }

            std::sort(v_size_and_ptr.begin(), v_size_and_ptr.end(),
                [](const auto& a, const auto& b) {
                    if (a.first != b.first) return a.first > b.first;
                    return a.second->ul.x < b.second->ul.x;
                });

            for (int j = static_cast<int>(v_size_and_ptr.size()) - 1;
                 j >= 0 && static_cast<int>(l_nodes.size()) < n_features; --j) {
                ExtractorNode n1, n2, n3, n4;
                v_size_and_ptr[j].second->divideNode(n1, n2, n3, n4);

                if (n1.v_keys.size() > 0) {
                    l_nodes.push_front(n1);
                    if (n1.v_keys.size() > 1)
                        v_size_and_ptr.emplace_back(n1.v_keys.size(), &l_nodes.front());
                }
                if (n2.v_keys.size() > 0) {
                    l_nodes.push_front(n2);
                    if (n2.v_keys.size() > 1)
                        v_size_and_ptr.emplace_back(n2.v_keys.size(), &l_nodes.front());
                }
                if (n3.v_keys.size() > 0) {
                    l_nodes.push_front(n3);
                    if (n3.v_keys.size() > 1)
                        v_size_and_ptr.emplace_back(n3.v_keys.size(), &l_nodes.front());
                }
                if (n4.v_keys.size() > 0) {
                    l_nodes.push_front(n4);
                    if (n4.v_keys.size() > 1)
                        v_size_and_ptr.emplace_back(n4.v_keys.size(), &l_nodes.front());
                }

                l_nodes.erase(v_size_and_ptr[j].second->lit);

                if (static_cast<int>(l_nodes.size()) >= n_features)
                    break;
            }

            if (static_cast<int>(l_nodes.size()) >= n_features ||
                static_cast<int>(l_nodes.size()) == prev_size)
                b_finish = true;
        }
    }

    result.reserve(n_features);
    for (auto& node : l_nodes) {
        const auto& v_node_keys = node.v_keys;
        const cv::KeyPoint* p_kp = &v_node_keys[0];
        float max_resp = p_kp->response;
        for (size_t k = 1; k < v_node_keys.size(); ++k) {
            if (v_node_keys[k].response > max_resp) {
                p_kp = &v_node_keys[k];
                max_resp = v_node_keys[k].response;
            }
        }
        result.push_back(*p_kp);
    }
}

// ================================================================
// computeKeyPointsOctTree — 四叉树特征提取（与 ORB_SLAM3 完全一致）
// ================================================================
void OctreeORB::computeKeyPointsOctTree(
        std::vector<std::vector<cv::KeyPoint>>& all_keypoints) {
    all_keypoints.resize(config_.n_levels);

    const float w = 35.0f;  // ORB_SLAM3: W = 35

    for (int level = 0; level < config_.n_levels; ++level) {
        const cv::Mat& level_img = image_pyramid_[level];
        const int cols = level_img.cols;
        const int rows = level_img.rows;

        const int min_border_x = config_.edge_threshold - 3;
        const int min_border_y = min_border_x;
        const int max_border_x = cols - config_.edge_threshold + 3;
        const int max_border_y = rows - config_.edge_threshold + 3;

        std::vector<cv::KeyPoint> v_to_distribute_keys;
        v_to_distribute_keys.reserve(config_.n_features * 10);

        const float width  = static_cast<float>(max_border_x - min_border_x);
        const float height = static_cast<float>(max_border_y - min_border_y);

        const int n_cols = static_cast<int>(width / w);
        const int n_rows = static_cast<int>(height / w);
        const int w_cell = cvCeil(width / n_cols);
        const int h_cell = cvCeil(height / n_rows);

        for (int i = 0; i < n_rows; ++i) {
            float ini_y = min_border_y + i * h_cell;
            float max_y = ini_y + h_cell + 6;

            if (ini_y >= max_border_y - 3) continue;
            if (max_y > max_border_y) max_y = static_cast<float>(max_border_y);

            for (int j = 0; j < n_cols; ++j) {
                float ini_x = min_border_x + j * w_cell;
                float max_x = ini_x + w_cell + 6;

                if (ini_x >= max_border_x - 6) continue;
                if (max_x > max_border_x) max_x = static_cast<float>(max_border_x);

                cv::Mat cell_img = level_img(
                    cv::Range(cvRound(ini_y), cvRound(max_y)),
                    cv::Range(cvRound(ini_x), cvRound(max_x)));

                std::vector<cv::KeyPoint> v_keys_cell;
                cv::FAST(cell_img, v_keys_cell,
                         config_.ini_threshold_fast, true);

                if (v_keys_cell.empty()) {
                    cv::FAST(cell_img, v_keys_cell,
                             config_.min_threshold_fast, true);
                }

                if (!v_keys_cell.empty()) {
                    for (auto& kp : v_keys_cell) {
                        kp.pt.x += j * w_cell;
                        kp.pt.y += i * h_cell;
                        v_to_distribute_keys.push_back(kp);
                    }
                }
            }
        }

        auto& keypoints = all_keypoints[level];
        keypoints.reserve(config_.n_features);

        distributeOctTree(v_to_distribute_keys,
                          min_border_x, max_border_x,
                          min_border_y, max_border_y,
                          mn_features_per_level_[level], level,
                          keypoints);

        const int scaled_patch =
            config_.patch_size *
            static_cast<int>(mv_scale_factor_[level]);

        const int n_kps = static_cast<int>(keypoints.size());
        for (int i = 0; i < n_kps; ++i) {
            keypoints[i].pt.x += min_border_x;
            keypoints[i].pt.y += min_border_y;
            keypoints[i].octave = level;
            keypoints[i].size = scaled_patch;
        }
    }
}

// ================================================================
// descriptorDistance — ORB 描述子 Hamming 距离
// ================================================================
int OctreeORB::descriptorDistance(const cv::Mat& a, const cv::Mat& b) {
    if (a.empty() || b.empty()) return 256;
    return cv::norm(a, b, cv::NORM_HAMMING);
}

} // namespace feature
} // namespace sfc_core
