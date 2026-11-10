/**
 * test_keyframe_policy.cpp
 *
 * KeyframePolicy 单元测试（纯逻辑，不需要任何相机数据）。
 *
 * 覆盖：
 *   - 健康跟踪 + 无运动：不插帧
 *   - 跟踪质量下降：过了最小间隔才插，间隔内不插
 *   - 内点绝对数够多时，比例下降不触发
 *   - 运动阈值触发 / 速度不可信时不触发
 *   - 强制间隔、兜底插入
 *   - ref_matches = 0 时比例分支不生效
 *
 * 编译后运行：
 *   ./test_keyframe_policy            # 正常跑，失败返回非 0
 *   ./test_keyframe_policy --verbose  # 打印每个用例的输入与判定
 *
 * 本文件不引入 gtest：仓库内测试统一是自带 main 的可执行文件。
 */

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "sfc_core/frontend/keyframe_policy.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const char* expr, int line) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::printf("    FAIL  line %d: %s\n", line, expr);
    }
}

#define CHECK(expr) check((expr), #expr, __LINE__)

using sfc_core::frontend::decideKeyframe;
using sfc_core::frontend::KeyframeDecision;
using sfc_core::frontend::KeyframeEvidence;
using sfc_core::frontend::KeyframePolicyParams;

// 一组"健康跟踪、静止"的基线证据，各用例在它上面改一个字段。
KeyframeEvidence healthyStill() {
    KeyframeEvidence ev;
    ev.frame_gap = 3;
    ev.inliers = 500;
    ev.ref_matches = 600;
    ev.velocity_valid = true;
    ev.translation_since_kf = 0.01;
    ev.rotation_since_kf = 0.005;
    return ev;
}

KeyframePolicyParams defaultParams() {
    return KeyframePolicyParams();
}

void report(const char* name, const KeyframeEvidence& ev, const KeyframeDecision& d) {
    std::printf("  %-42s need=%d forced=%d rescue=%d quality=%d motion=%d  "
                "(gap=%d in=%d ref=%d t=%.3f r=%.3f)\n",
                name, d.need_keyframe ? 1 : 0, d.forced ? 1 : 0, d.rescue ? 1 : 0,
                d.by_tracking_quality ? 1 : 0, d.by_motion ? 1 : 0,
                ev.frame_gap, ev.inliers, ev.ref_matches,
                ev.translation_since_kf, ev.rotation_since_kf);
}

// 用例 1：跟踪健康、几乎不动 —— 不该插帧。
void testHealthyStill() {
    std::printf("[1] 健康跟踪 + 静止\n");
    const KeyframeDecision d = decideKeyframe(defaultParams(), healthyStill());
    report("healthy_still", healthyStill(), d);
    CHECK(!d.need_keyframe);
}

// 用例 2：跟踪质量下降，但还在最小间隔内 —— 不插。
void testQualityInsideGap() {
    std::printf("[2] 质量下降但未过最小间隔\n");
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 3;          // < min_gap_frames(8)
    ev.inliers = 60;
    ev.ref_matches = 600;      // 60 < 600*0.30 = 180 → 比例低
    const KeyframeDecision d = decideKeyframe(defaultParams(), ev);
    report("quality_inside_gap", ev, d);
    CHECK(d.by_tracking_quality);
    CHECK(!d.need_keyframe);
}

// 用例 3：同样的质量下降，过了最小间隔 —— 插，且标记来自质量分支。
void testQualityOutsideGap() {
    std::printf("[3] 质量下降且过了最小间隔\n");
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 12;
    ev.inliers = 60;
    ev.ref_matches = 600;
    const KeyframeDecision d = decideKeyframe(defaultParams(), ev);
    report("quality_outside_gap", ev, d);
    CHECK(d.need_keyframe);
    CHECK(d.by_tracking_quality);
    CHECK(!d.forced);
}

// 用例 4：比例低但内点绝对数够多 —— 不插（绝对下限保护）。
void testAbsoluteFloorProtects() {
    std::printf("[4] 内点绝对值充足，比例下降不算退化\n");
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 20;
    ev.inliers = 200;
    ev.ref_matches = 2000;     // 200 < 600 → 比例低；但 200 >= min_tracked_inliers(120)
    const KeyframeDecision d = decideKeyframe(defaultParams(), ev);
    report("absolute_floor", ev, d);
    CHECK(!d.by_tracking_quality);
    CHECK(!d.need_keyframe);
}

// 用例 5：运动超阈值 —— 插。
void testMotionTriggers() {
    std::printf("[5] 平移超过关键帧阈值\n");
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 10;
    ev.translation_since_kf = 0.25;   // >= keyframe_translation(0.15)
    const KeyframeDecision d = decideKeyframe(defaultParams(), ev);
    report("motion_translation", ev, d);
    CHECK(d.need_keyframe);
    CHECK(d.by_motion);
}

// 用例 6：只有旋转超阈值也插。
void testRotationTriggers() {
    std::printf("[6] 旋转超过关键帧阈值\n");
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 10;
    ev.rotation_since_kf = 0.30;      // >= keyframe_rotation(0.05)
    const KeyframeDecision d = decideKeyframe(defaultParams(), ev);
    report("motion_rotation", ev, d);
    CHECK(d.need_keyframe);
    CHECK(d.by_motion);
}

// 用例 7：速度不可信时不看运动阈值（否则把预测误差当真实位移）。
void testMotionNeedsValidVelocity() {
    std::printf("[7] 速度不可信 → 运动分支不生效\n");
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 10;
    ev.translation_since_kf = 0.80;
    ev.velocity_valid = false;
    const KeyframeDecision d = decideKeyframe(defaultParams(), ev);
    report("motion_invalid_velocity", ev, d);
    CHECK(!d.by_motion);
    CHECK(!d.need_keyframe);
}

// 用例 8：强制间隔到点 —— 插，且优先于最小间隔。
void testForcedInterval() {
    std::printf("[8] 到达强制插入间隔\n");
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 30;         // == forced_interval
    const KeyframeDecision d = decideKeyframe(defaultParams(), ev);
    report("forced_interval", ev, d);
    CHECK(d.forced);
    CHECK(d.need_keyframe);
}

// 用例 9：强制间隔为 0 表示关闭。
void testForcedDisabled() {
    std::printf("[9] forced_interval = 0 关闭强制插入\n");
    KeyframePolicyParams p = defaultParams();
    p.forced_interval = 0;
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 200;
    const KeyframeDecision d = decideKeyframe(p, ev);
    report("forced_disabled", ev, d);
    CHECK(!d.forced);
    CHECK(!d.need_keyframe);
}

// 用例 10：长期内点极少 —— 兜底插入。
void testRescue() {
    std::printf("[10] 长期内点极少 → 兜底插入\n");
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 61;         // >= rescue_max_gap(60)
    ev.inliers = 20;           // <= rescue_min_inliers(40)
    ev.ref_matches = 700;
    const KeyframeDecision d = decideKeyframe(defaultParams(), ev);
    report("rescue", ev, d);
    CHECK(d.rescue);
    CHECK(d.need_keyframe);
}

// 用例 11：兜底也要有间隔，刚插过不重复插。
// 把 ref_matches 置 0 是为了关掉"质量下降"分支 —— 否则那条会独立触发，
// 测不到兜底分支自己的间隔条件。
void testRescueNeedsGap() {
    std::printf("[11] 兜底仍需间隔\n");
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 20;         // < rescue_max_gap
    ev.inliers = 20;
    ev.ref_matches = 0;
    const KeyframeDecision d = decideKeyframe(defaultParams(), ev);
    report("rescue_needs_gap", ev, d);
    CHECK(!d.rescue);
    CHECK(!d.need_keyframe);
}

// 用例 12：ref_matches = 0（参考帧没有活着的点）时不因比例插帧。
void testNoRefMatches() {
    std::printf("[12] ref_matches = 0 → 比例分支不生效\n");
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 12;
    ev.inliers = 10;
    ev.ref_matches = 0;
    const KeyframeDecision d = decideKeyframe(defaultParams(), ev);
    report("no_ref_matches", ev, d);
    CHECK(!d.by_tracking_quality);
    CHECK(!d.need_keyframe);
}

// 用例 13：最小间隔参数非法（0）时按 1 处理，不会退化成同帧重复插帧。
void testDegenerateMinGap() {
    std::printf("[13] min_gap_frames = 0 按 1 处理\n");
    KeyframePolicyParams p = defaultParams();
    p.min_gap_frames = 0;
    p.forced_interval = 0;
    KeyframeEvidence ev = healthyStill();
    ev.frame_gap = 0;
    ev.translation_since_kf = 5.0;   // 极端位移，但 gap=0
    const KeyframeDecision d = decideKeyframe(p, ev);
    report("degenerate_min_gap", ev, d);
    CHECK(!d.need_keyframe);
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    std::printf("==== test_keyframe_policy ====\n");

    testHealthyStill();
    testQualityInsideGap();
    testQualityOutsideGap();
    testAbsoluteFloorProtects();
    testMotionTriggers();
    testRotationTriggers();
    testMotionNeedsValidVelocity();
    testForcedInterval();
    testForcedDisabled();
    testRescue();
    testRescueNeedsGap();
    testNoRefMatches();
    testDegenerateMinGap();

    std::printf("==== %d checks, %d failures ====\n", g_checks, g_failures);
    if (g_failures != 0) {
        std::printf("RESULT: FAILED\n");
        return 1;
    }
    std::printf("RESULT: PASSED\n");
    return 0;
}
