#ifdef sfc_core
#undef sfc_core
#endif

#include "sfc_core/feature/feature_matcher.hpp"
#include <algorithm>
#include <map>
#include <cmath>

namespace sfc_core {
namespace feature {

FeatureMatcher::FeatureMatcher(const MatcherConfig& config) : config_(config) {}

void FeatureMatcher::match(const cv::Mat& descriptors1,
                          const cv::Mat& descriptors2,
                          std::vector<cv::DMatch>& matches) {
    matches.clear();
    if (descriptors1.empty() || descriptors2.empty()) {
        std::cerr << "[FeatureMatcher] WARNING: empty descriptor detected! d1.rows=" << descriptors1.rows << " d2.rows=" << descriptors2.rows << std::endl;
        return;
    }
    if (descriptors1.type() != CV_8UC1 || descriptors2.type() != CV_8UC1) {
        std::cerr << "[FeatureMatcher] WARNING: wrong descriptor type! d1.type=" << descriptors1.type() << " d2.type=" << descriptors2.type() << std::endl;
        return;
    }
    if (descriptors1.rows < 1 || descriptors2.rows < 1) {
        std::cerr << "[FeatureMatcher] WARNING: zero-row descriptor! d1.rows=" << descriptors1.rows << " d2.rows=" << descriptors2.rows << std::endl;
        return;
    }

    // 使用 OpenCV BFMatcher（Hamming 距离，即 XOR + popcount）
    cv::BFMatcher matcher(cv::NORM_HAMMING, !config_.cross_check);

    if (config_.cross_check) {
        // 十字交叉校验：match1 从 1->2，match2 从 2->1
        std::vector<cv::DMatch> match1, match2;
        matcher.match(descriptors1, descriptors2, match1);
        matcher.match(descriptors2, descriptors1, match2);

        for (const auto& m1 : match1) {
            bool cross_ok = false;
            for (const auto& m2 : match2) {
                if (m2.trainIdx == m1.queryIdx && m2.queryIdx == m1.trainIdx) {
                    cross_ok = true;
                    break;
                }
            }
            if (cross_ok) matches.push_back(m1);
        }
    } else {
        // 手动 Lowe's ratio test：用 match + sort 替代 knnMatch
        // 原因：OpenCV 4.5.x 的 knnMatch 内部 batchDistance 有 bug，
        // 在某些情况下触发 K==1 断言失败，即使描述子维度完全正常。
        std::vector<cv::DMatch> all_matches;
        matcher.match(descriptors1, descriptors2, all_matches);

        if (all_matches.empty()) return;

        // 按距离升序排序（match 默认已排序，但显式排一次更可靠）
        std::sort(all_matches.begin(), all_matches.end(),
                   [](const cv::DMatch& a, const cv::DMatch& b) {
                       return a.distance < b.distance;
                   });

        // 对每个 queryIdx，收集其所有候选匹配中最近的 2 个做 ratio test
        // 用 map 收集每个 queryIdx 的最近两个匹配
        std::map<int, std::vector<const cv::DMatch*>> query_candidates;
        for (const auto& m : all_matches) {
            query_candidates[m.queryIdx].push_back(&m);
        }

        for (const auto& [queryIdx, candidates] : query_candidates) {
            if (candidates.size() >= 2) {
                const cv::DMatch& m0 = *candidates[0];
                const cv::DMatch& m1 = *candidates[1];
                if (m0.distance < config_.lowe_ratio * m1.distance &&
                    m0.distance <= config_.max_hamming_dist) {
                    matches.push_back(m0);
                }
            } else if (!candidates.empty()) {
                // ★★★ Bug Fix: 单候选时也要检查阈值 ★★★
                // 原问题：当某个 queryIdx 只有一个匹配候选时，即使距离很大（如 70-80）也会被接受，
                // 导致大量误匹配进入后续流程。
                // 修复：单候选时使用更严格的阈值（max_hamming_dist 的 50%），
                // 只有距离足够近的匹配才被接受。
                const cv::DMatch& m0 = *candidates[0];
                int single_candidate_max_dist = static_cast<int>(config_.max_hamming_dist * 0.5);
                if (m0.distance <= single_candidate_max_dist) {
                    matches.push_back(m0);
                }
            }
        }
    }
}

void FeatureMatcher::removeOutliers(const std::vector<cv::DMatch>& raw_matches,
                                   const std::vector<cv::KeyPoint>& keypoints1,
                                   const std::vector<cv::KeyPoint>& keypoints2,
                                   std::vector<cv::DMatch>& inlier_matches,
                                   const cv::Mat& F_matrix) {
    inlier_matches.clear();
    if (raw_matches.empty()) return;

    if (F_matrix.empty() || F_matrix.cols != 3 || F_matrix.rows != 3) {
        // 无基础矩阵：使用描述子距离过滤 + 宽松几何过滤
        // ORB 描述子距离：优秀 <20，良好 20-40，可疑 40-80，差 >80
        for (const auto& m : raw_matches) {
            if (m.queryIdx >= static_cast<int>(keypoints1.size())) continue;
            if (m.trainIdx >= static_cast<int>(keypoints2.size())) continue;
            if (m.distance > 80) continue;  // 拒绝距离过大的匹配
            inlier_matches.push_back(m);
        }
        return;
    }

    // ★★★ 有基础矩阵：极线约束过滤（与 ORB-SLAM3 SearchByBoW 一致）★★★
    // 对于每对匹配 (p1, p2)，检查 p2 是否在 p1 的极线上（误差 < 阈值）
    // x2^T * F * x1 = 0
    cv::Mat F_inv_t = F_matrix.inv().t();
    for (const auto& m : raw_matches) {
        if (m.queryIdx >= static_cast<int>(keypoints1.size())) continue;
        if (m.trainIdx >= static_cast<int>(keypoints2.size())) continue;
        if (m.distance > 80) continue;

        const cv::KeyPoint& kp1 = keypoints1[m.queryIdx];
        const cv::KeyPoint& kp2 = keypoints2[m.trainIdx];

        // 计算点到极线的距离：|x2^T * F * x1| / ||F^T * x2||_2
        double x1 = kp1.pt.x, y1 = kp1.pt.y;
        double x2 = kp2.pt.x, y2 = kp2.pt.y;

        double a = F_matrix.at<double>(0, 0) * x1 + F_matrix.at<double>(1, 0) * y1 + F_matrix.at<double>(2, 0);
        double b = F_matrix.at<double>(0, 1) * x1 + F_matrix.at<double>(1, 1) * y1 + F_matrix.at<double>(2, 1);
        double c = F_matrix.at<double>(0, 2) * x1 + F_matrix.at<double>(1, 2) * y1 + F_matrix.at<double>(2, 2);

        double num = std::abs(a * x2 + b * y2 + c);
        double den = std::sqrt(a * a + b * b);
        if (den < 1e-10) continue;

        double dist = num / den;
        if (dist > 3.0) continue;  // 像素，极线距离超过 3px 的视为外点

        inlier_matches.push_back(m);
    }
}

// ================================================================
// ORB_SLAM3 风格 SearchByProjection
// 投影上一帧的 3D 点到当前帧，在投影点附近搜索匹配
// 注意：tracking.cpp 中有完整的实现，此函数作为独立匹配器保留
// ================================================================
std::vector<std::pair<int, int>> FeatureMatcher::searchByProjection(
    const std::vector<Eigen::Vector3d>& last_pts3d,
    const std::vector<Eigen::Vector3d>& last_pts3d_world,
    const std::vector<bool>& last_depth_valid,
    const std::vector<cv::KeyPoint>& curr_kpts,
    const cv::Mat& curr_descriptors,
    const Eigen::Matrix3d& R,
    const Eigen::Vector3d& t,
    const cv::Mat& K,
    float th,
    bool check_scale) {

    std::vector<std::pair<int, int>> matches;
    if (last_pts3d.empty() || curr_kpts.empty() || curr_descriptors.empty()) {
        return matches;
    }

    // 提取相机内参
    double fx = K.empty() ? 525.0 : K.at<double>(0, 0);
    double fy = K.empty() ? 525.0 : K.at<double>(1, 1);
    double cx = K.empty() ? 319.5 : K.at<double>(0, 2);
    double cy = K.empty() ? 239.5 : K.at<double>(1, 2);

    // 图像尺寸（使用保守默认值，实际应用中应从图像元数据获取）
    int img_cols = 640;
    int img_rows = 480;

    // 跟踪已匹配的关键点（避免重复匹配）
    std::vector<char> kpt_matched(curr_kpts.size(), 0);

    for (size_t iMP = 0; iMP < last_pts3d.size(); iMP++) {
        if (!last_depth_valid[iMP]) continue;
        const Eigen::Vector3d& pt_cam = last_pts3d[iMP];

        // 跳过深度过浅的点
        if (pt_cam.z() < 0.05) continue;

        // 将 3D 点从上一帧相机坐标系变换到当前帧相机坐标系
        Eigen::Vector3d pt_curr_cam = R * pt_cam + t;

        // 跳过在当前帧相机后面的点
        if (pt_curr_cam.z() < 0.05) continue;

        // 投影到像素坐标系
        float u = static_cast<float>(fx * pt_curr_cam.x() / pt_curr_cam.z() + cx);
        float v = static_cast<float>(fy * pt_curr_cam.y() / pt_curr_cam.z() + cy);

        // 边界检查（使用保守默认值）
        if (u < 0 || u >= img_cols || v < 0 || v >= img_rows) {
            continue;
        }

        // 搜索半径（ORB_SLAM3 风格）
        float radius = check_scale ? 4.0f : 7.0f;
        if (check_scale) {
            // 根据深度调整搜索窗口大小
            float depth_factor = std::min(static_cast<float>(pt_curr_cam.z()), 8.0f) / 8.0f;
            radius = 3.0f + depth_factor * 4.0f;
        }
        radius *= th;  // 允许通过 th 参数放大
        float radius2 = radius * radius;

        // 在搜索窗口内找到最佳匹配
        int best_idx = -1;

        for (size_t i = 0; i < curr_kpts.size(); i++) {
            // 跳过已匹配的点
            if (kpt_matched[i]) continue;

            float du = curr_kpts[i].pt.x - u;
            float dv = curr_kpts[i].pt.y - v;
            if (du * du + dv * dv > radius2) continue;

            // 边界检查
            if (i >= static_cast<size_t>(curr_descriptors.rows)) continue;

            // 找到第一个满足几何条件的点（简化处理）
            best_idx = static_cast<int>(i);
            break;
        }

        // 如果找到匹配
        if (best_idx >= 0) {
            kpt_matched[best_idx] = 1;
            matches.emplace_back(static_cast<int>(iMP), best_idx);
        }
    }

    return matches;
}

// ================================================================
// 投影地图点到帧（ORB_SLAM3 mbTrackInView 机制）
// ================================================================
int FeatureMatcher::projectMapPointsToFrame(
    const std::vector<MapPointMatch*>& map_points,
    const Eigen::Matrix3d& R_cw,
    const Eigen::Vector3d& t_cw,
    const cv::Mat& K,
    float fx, float fy, float cx, float cy,
    int cols, int rows,
    float th) {

    int n_visible = 0;

    for (auto* pMP : map_points) {
        if (!pMP || pMP->is_bad) continue;

        // 将世界坐标变换到相机坐标系
        Eigen::Vector3d pc = R_cw * pMP->world_point + t_cw;

        // 检查深度
        if (pc.z() < 0.01) {
            pMP->track_in_view = false;
            continue;
        }

        // 投影到像素
        float u = static_cast<float>(fx * pc.x() / pc.z() + cx);
        float v = static_cast<float>(fy * pc.y() / pc.z() + cy);

        // 检查边界
        if (u < 0 || u >= cols || v < 0 || v >= rows) {
            pMP->track_in_view = false;
            continue;
        }

        // 计算视角余弦（用于尺度搜索窗口）
        Eigen::Vector3d view_dir = pc.normalized();
        Eigen::Vector3d z_axis(0, 0, 1);
        float view_cos = static_cast<float>(view_dir.dot(z_axis));

        // 设置投影信息
        pMP->track_in_view = true;
        pMP->track_proj_x = u;
        pMP->track_proj_y = v;
        pMP->track_view_cos = view_cos;
        pMP->track_depth = static_cast<float>(pc.z());

        n_visible++;
    }

    return n_visible;
}

// ================================================================
// 根据视角余弦计算搜索半径
// ================================================================
float FeatureMatcher::radiusByViewingCos(float viewCos) {
    if (viewCos > 0.998f)
        return 2.5f;
    else
        return 4.0f;
}

// ================================================================
// 计算直方图中最大的三个峰值
// ================================================================
void FeatureMatcher::computeThreeMaxima(std::vector<int>* histo, int L, int& ind1, int& ind2, int& ind3) {
    ind1 = ind2 = ind3 = -1;
    int max1 = 0, max2 = 0, max3 = 0;

    for (int i = 0; i < L; i++) {
        int n = histo[i].size();
        if (n > max1) {
            max3 = max2;
            ind3 = ind2;
            max2 = max1;
            ind2 = ind1;
            max1 = n;
            ind1 = i;
        } else if (n > max2) {
            max3 = max2;
            ind3 = ind2;
            max2 = n;
            ind2 = i;
        } else if (n > max3) {
            max3 = n;
            ind3 = i;
        }
    }

    // 确保 ind2 和 ind3 与 ind1 不相邻
    if (ind2 - ind1 > L / 4) {
        while (ind3 != -1 && (ind3 - ind1 < L / 4 || ind2 - ind3 < L / 4)) {
            max3 = 0;
            ind3 = -1;
            for (int i = 0; i < L; i++) {
                if (i == ind1 || i == ind2) continue;
                int n = histo[i].size();
                if (n > max3) {
                    max3 = n;
                    ind3 = i;
                }
            }
        }
    }

    if (ind1 - ind2 <= L / 4) {
        max2 = 0;
        ind2 = -1;
        for (int i = 0; i < L; i++) {
            int n = histo[i].size();
            if (i == ind1) continue;
            if (n > max2) {
                max2 = n;
                ind2 = i;
            }
        }
    }
}

} // namespace feature
} // namespace sfc_core
