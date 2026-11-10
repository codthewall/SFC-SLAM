/**
 * test_reprojection_association.cpp
 *
 * 投影关联判据的单元测试（纯逻辑，不需要相机数据与 g2o）。
 *
 * 覆盖：
 *   - findKeypointNearProjection：按描述子而非像素距离取点
 *   - 窗内两个候选"一样像"时按 Lowe 比值拒绝
 *   - 最近邻超过汉明门限 / 落在半径外 / 已被占用 → 不匹配
 *   - verifyAssociationsAgainstPose：与位姿自洽的关联保留，不自洽的丢弃
 *   - 同一 keypoint 只保留误差最小的那条关联
 *   - 投影到相机后方的点被丢弃
 *   - matchReferenceByProjection：投影落在画面外则整条链路不产出匹配
 *
 * 编译后运行：
 *   ./test_reprojection_association            # 失败返回非 0
 *   ./test_reprojection_association --verbose  # 打印每个用例的判定
 *
 * 本文件不引入 gtest：仓库内测试统一是自带 main 的可执行文件。
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "sfc_core/frontend/reprojection_association.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;
bool g_verbose = false;

void check(bool cond, const char* expr, int line) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::printf("    FAIL  line %d: %s\n", line, expr);
    } else if (g_verbose) {
        std::printf("    ok    line %d: %s\n", line, expr);
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

/// 32 字节全零描述子（Hamming 距离的基准）。
cv::Mat makeZeroDescriptors(int rows) {
    return cv::Mat(rows, 32, CV_8U, cv::Scalar(0));
}

/// 让第 row 行的第 byte_index 个字节等于 value，其余字节为 0。
/// 这样与全零描述子的汉明距离恰好等于 popcount(value)，用例可以精确指定距离。
void setByte(cv::Mat& descriptors, int row, int byte_index, unsigned char value) {
    std::memset(descriptors.ptr(row), 0, 32);
    descriptors.ptr(row)[byte_index] = value;
}

std::vector<cv::KeyPoint> makeKeypoints(const std::vector<cv::Point2f>& pts) {
    std::vector<cv::KeyPoint> kps;
    kps.reserve(pts.size());
    for (const auto& p : pts) kps.emplace_back(p, 1.0f);
    return kps;
}

sfc_core::frontend::ProjectionSearchConfig defaultSearchConfig() {
    sfc_core::frontend::ProjectionSearchConfig cfg;
    cfg.radius_px = 15.0f;
    cfg.hamming_threshold = 60;
    cfg.lowe_ratio = 0.75f;
    return cfg;
}

// ---------------------------------------------------------------
// findKeypointNearProjection：取点顺序必须是"描述子优先，像素仅用于圈定窗"
// ---------------------------------------------------------------
void testDescriptorBeatsPixelProximity() {
    std::printf("[case] 描述子优先于像素距离\n");
    // 参考描述子全零；投影点 (100,100)。
    cv::Mat ref_desc = makeZeroDescriptors(1);
    // kp0 像素上就在投影点上（距离 0），但描述子差 4 bit；
    // kp1 偏离 3 px，描述子完全一致（距离 0）。
    // 旧实现只取"投影位置最近的像素"→ 选 kp0，把错配喂给 PnP。
    cv::Mat curr_desc = makeZeroDescriptors(2);
    setByte(curr_desc, 0, 0, 0x0F);  // popcount 4
    setByte(curr_desc, 1, 0, 0x00);  // 距离 0
    const auto kps = makeKeypoints({cv::Point2f(100.0f, 100.0f),
                                    cv::Point2f(103.0f, 100.0f)});
    const std::vector<char> blocked(kps.size(), 0);

    const auto m = sfc_core::frontend::findKeypointNearProjection(
        kps, curr_desc, ref_desc, cv::Point2f(100.0f, 100.0f),
        defaultSearchConfig(), blocked);
    CHECK(m.current_index == 1);
    CHECK(m.descriptor_distance == 0);
}

void testLoweRatioRejectsAmbiguous() {
    std::printf("[case] 窗内两个候选一样像 → 按比值拒绝\n");
    cv::Mat ref_desc = makeZeroDescriptors(1);
    // 两个候选描述子距离都是 2（0x03 的 popcount = 2）。
    cv::Mat curr_desc = makeZeroDescriptors(2);
    setByte(curr_desc, 0, 0, 0x03);
    setByte(curr_desc, 1, 0, 0x03);
    const auto kps = makeKeypoints({cv::Point2f(100.0f, 100.0f),
                                    cv::Point2f(102.0f, 100.0f)});
    const std::vector<char> blocked(kps.size(), 0);

    const auto m = sfc_core::frontend::findKeypointNearProjection(
        kps, curr_desc, ref_desc, cv::Point2f(101.0f, 100.0f),
        defaultSearchConfig(), blocked);
    CHECK(m.current_index == -1);
}

void testHammingThresholdAndRadiusAndBlocked() {
    std::printf("[case] 汉明门限 / 搜索半径 / 占用标记\n");
    cv::Mat ref_desc = makeZeroDescriptors(1);
    cv::Mat curr_desc = makeZeroDescriptors(2);
    setByte(curr_desc, 0, 0, 0xFF);  // 距离 128，超过门限 60
    setByte(curr_desc, 1, 0, 0x00);  // 距离 0，但远在半径之外
    const auto kps = makeKeypoints({cv::Point2f(100.0f, 100.0f),
                                    cv::Point2f(400.0f, 400.0f)});
    std::vector<char> blocked(kps.size(), 0);
    const auto cfg = defaultSearchConfig();

    const auto far_and_weak = sfc_core::frontend::findKeypointNearProjection(
        kps, curr_desc, ref_desc, cv::Point2f(100.0f, 100.0f), cfg, blocked);
    CHECK(far_and_weak.current_index == -1);

    // 只留半径内那个（距离 128 的那个），仍然超门限。
    blocked[1] = 1;
    const auto over_threshold = sfc_core::frontend::findKeypointNearProjection(
        kps, curr_desc, ref_desc, cv::Point2f(100.0f, 100.0f), cfg, blocked);
    CHECK(over_threshold.current_index == -1);

    // 未被占用时，半径内的近距离候选应当匹配上。
    cv::Mat good_desc = makeZeroDescriptors(1);
    setByte(good_desc, 0, 0, 0x01);  // 距离 1
    const auto one_kp = makeKeypoints({cv::Point2f(100.0f, 100.0f)});
    const std::vector<char> free_mark(1, 0);
    const auto ok = sfc_core::frontend::findKeypointNearProjection(
        one_kp, good_desc, ref_desc, cv::Point2f(100.0f, 100.0f), cfg, free_mark);
    CHECK(ok.current_index == 0);

    const std::vector<char> taken(1, 1);
    const auto blocked_out = sfc_core::frontend::findKeypointNearProjection(
        one_kp, good_desc, ref_desc, cv::Point2f(100.0f, 100.0f), cfg, taken);
    CHECK(blocked_out.current_index == -1);
}

// ---------------------------------------------------------------
// verifyAssociationsAgainstPose
// ---------------------------------------------------------------
sfc_core::frontend::PinholeIntrinsics testIntrinsics() {
    sfc_core::frontend::PinholeIntrinsics intr;
    intr.fx = 500.0;
    intr.fy = 500.0;
    intr.cx = 320.0;
    intr.cy = 240.0;
    return intr;
}

void testConsistentAssociationsKept() {
    std::printf("[case] 与位姿自洽的关联保留，不自洽的丢弃\n");
    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d t = Eigen::Vector3d::Zero();
    // 参考帧：两个前方 2 m 的点（相机系 = 世界系）。
    std::vector<Eigen::Vector3d> pts_cam = {
        Eigen::Vector3d(0.0, 0.0, 2.0),   // 投到 (320,240)
        Eigen::Vector3d(0.0, 0.0, 2.0)};
    const std::vector<bool> depth_valid = {true, true};
    // 当前帧：第 0 个观测落在投影点上；第 1 个偏 10 px。
    const std::vector<cv::Point2f> current_uv = {
        cv::Point2f(320.0f, 240.0f),
        cv::Point2f(330.0f, 240.0f)};
    std::vector<cv::DMatch> matches;
    matches.emplace_back(0, 0, 1.0f);
    matches.emplace_back(1, 1, 1.0f);

    const auto assoc = sfc_core::frontend::verifyAssociationsAgainstPose(
        matches, pts_cam, depth_valid, R, t, current_uv, R, t, testIntrinsics(),
        6.0f, 0.05);
    CHECK(assoc.size() == 1);
    if (assoc.size() == 1) {
        CHECK(assoc[0].current_index == 0);
        CHECK(assoc[0].ref_index == 0);
    }
}

void testMutualExclusionKeepsSmallestError() {
    std::printf("[case] 同一 keypoint 只保留误差最小的关联\n");
    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d t = Eigen::Vector3d::Zero();
    // 两个世界点投影到略有差异的位置，但都指向同一个当前帧 keypoint。
    std::vector<Eigen::Vector3d> pts_cam = {
        Eigen::Vector3d(0.004, 0.0, 2.0),  // 投到 (321,240)，误差 1 px
        Eigen::Vector3d(0.000, 0.0, 2.0)};  // 投到 (320,240)，误差 0 px
    const std::vector<bool> depth_valid = {true, true};
    const std::vector<cv::Point2f> current_uv = {cv::Point2f(320.0f, 240.0f)};
    std::vector<cv::DMatch> matches;
    matches.emplace_back(0, 0, 1.0f);
    matches.emplace_back(1, 0, 1.0f);

    const auto assoc = sfc_core::frontend::verifyAssociationsAgainstPose(
        matches, pts_cam, depth_valid, R, t, current_uv, R, t, testIntrinsics(),
        6.0f, 0.05);
    CHECK(assoc.size() == 1);
    if (assoc.size() == 1) {
        CHECK(assoc[0].ref_index == 1);
        CHECK(std::abs(assoc[0].world_point.z() - 2.0) < 1e-9);
    }
}

void testBehindCameraAndInvalidDepthDropped() {
    std::printf("[case] 相机后方 / 深度无效的点被丢弃\n");
    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d t = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> pts_cam = {
        Eigen::Vector3d(0.0, 0.0, -2.0),  // 相机后方
        Eigen::Vector3d(0.0, 0.0, 2.0)};  // 深度标记为无效
    const std::vector<bool> depth_valid = {true, false};
    const std::vector<cv::Point2f> current_uv = {
        cv::Point2f(320.0f, 240.0f),
        cv::Point2f(320.0f, 240.0f)};
    std::vector<cv::DMatch> matches;
    matches.emplace_back(0, 0, 1.0f);
    matches.emplace_back(1, 1, 1.0f);

    const auto assoc = sfc_core::frontend::verifyAssociationsAgainstPose(
        matches, pts_cam, depth_valid, R, t, current_uv, R, t, testIntrinsics(),
        6.0f, 0.05);
    CHECK(assoc.empty());
}

// ---------------------------------------------------------------
// matchReferenceByProjection：重定位"策略 1"整条链路
// ---------------------------------------------------------------
void testProjectionPipeline() {
    std::printf("[case] 参考点按预测位姿投影后取描述子一致的 keypoint\n");
    const Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    const Eigen::Vector3d t = Eigen::Vector3d::Zero();

    const auto ref_kps = makeKeypoints({cv::Point2f(320.0f, 240.0f)});
    cv::Mat ref_desc = makeZeroDescriptors(1);
    std::vector<Eigen::Vector3d> ref_pts = {Eigen::Vector3d(0.0, 0.0, 2.0)};
    const std::vector<bool> ref_depth = {true};

    cv::Mat curr_desc = makeZeroDescriptors(2);
    setByte(curr_desc, 0, 0, 0x03);  // 距离 2
    setByte(curr_desc, 1, 0, 0x01);  // 距离 1，应被选中
    const auto curr_kps = makeKeypoints({cv::Point2f(321.0f, 240.0f),
                                         cv::Point2f(318.0f, 239.0f)});

    auto cfg = defaultSearchConfig();
    const auto matches = sfc_core::frontend::matchReferenceByProjection(
        ref_kps, ref_desc, ref_pts, ref_depth, R, t, R, t, curr_kps, curr_desc, 640,
        480, testIntrinsics(), cfg);
    CHECK(matches.size() == 1);
    if (matches.size() == 1) {
        CHECK(matches[0].queryIdx == 0);
        CHECK(matches[0].trainIdx == 1);
    }

    // 预测位姿把点推到画面外（u≈1070 > 640），一条匹配都不该产生。
    const Eigen::Vector3d t_far(-3.0, 0.0, 0.0);
    const auto outside = sfc_core::frontend::matchReferenceByProjection(
        ref_kps, ref_desc, ref_pts, ref_depth, R, t, R, t_far, curr_kps, curr_desc, 640,
        480, testIntrinsics(), cfg);
    CHECK(outside.empty());
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--verbose") g_verbose = true;
    }

    testDescriptorBeatsPixelProximity();
    testLoweRatioRejectsAmbiguous();
    testHammingThresholdAndRadiusAndBlocked();
    testConsistentAssociationsKept();
    testMutualExclusionKeepsSmallestError();
    testBehindCameraAndInvalidDepthDropped();
    testProjectionPipeline();

    std::printf("\n%d checks, %d failure(s)\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("FAILED\n");
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
