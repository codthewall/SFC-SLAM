// ================================================================
// 关键帧插入策略实现
// ================================================================
// 纯逻辑，不依赖 OpenCV / Eigen / ROS，便于单测与复用。
// 参数语义与判定顺序见 include/sfc_core/frontend/keyframe_policy.hpp。
// ================================================================
#include "sfc_core/frontend/keyframe_policy.hpp"

#include <algorithm>

namespace sfc_core {
namespace frontend {

namespace {

// 判定 1：强制插入。间隔到点就插，与跟踪质量无关。
bool isForced(const KeyframePolicyParams& params, int gap) {
    if (params.forced_interval <= 0) {
        return false;
    }
    return gap >= params.forced_interval;
}

// 判定 2：兜底插入。内点数长期偏低说明局部地图已经没什么可用点，
// 靠"质量下降"分支（要求 ref_matches > 0 且相对比例）不一定能触发，
// 因此单独留一条只按绝对内点数和间隔的逃生通道。
bool isRescue(const KeyframePolicyParams& params, const KeyframeEvidence& ev, int gap) {
    if (params.rescue_max_gap <= 0) {
        return false;
    }
    return ev.inliers <= params.rescue_min_inliers && gap >= params.rescue_max_gap;
}

// 判定 3：跟踪质量下降。
// 只在参考点足够多时才看相对比例 —— ref_matches 很小时比例没有统计意义。
bool isTrackingQualityLow(const KeyframePolicyParams& params, const KeyframeEvidence& ev) {
    if (ev.ref_matches <= 0) {
        return false;
    }
    const double ratio_thresh = ev.ref_matches * params.weak_tracked_ratio;
    const bool ratio_low = static_cast<double>(ev.inliers) < ratio_thresh;
    if (!ratio_low) {
        return false;
    }
    // 内点绝对数够多时，比例下降只是场景变宽，不值得插帧。
    return ev.inliers < params.min_tracked_inliers;
}

// 判定 4：运动足够大。速度不可信时不看运动阈值（否则会把预测误差当真实位移）。
bool isMotionLarge(const KeyframePolicyParams& params, const KeyframeEvidence& ev) {
    if (!ev.velocity_valid) {
        return false;
    }
    return ev.translation_since_kf >= params.keyframe_translation ||
           ev.rotation_since_kf >= params.keyframe_rotation;
}

}  // namespace

KeyframeDecision decideKeyframe(const KeyframePolicyParams& params,
                                const KeyframeEvidence& evidence) {
    KeyframeDecision decision;

    const int gap = std::max(0, evidence.frame_gap);
    // 最小间隔至少要能挡住"同一帧插两次"，最小值取 1。
    const int min_gap = std::max(1, params.min_gap_frames);

    decision.forced = isForced(params, gap);
    decision.rescue = isRescue(params, evidence, gap);
    decision.by_tracking_quality = isTrackingQualityLow(params, evidence);
    decision.by_motion = isMotionLarge(params, evidence);

    // 强制与兜底本身就带着大间隔（forced_interval / rescue_max_gap），
    // 不需要再过最小间隔这一关；质量与运动两条必须过。
    if (decision.forced || decision.rescue) {
        decision.need_keyframe = true;
        return decision;
    }

    if (gap < min_gap) {
        return decision;
    }

    decision.need_keyframe = decision.by_tracking_quality || decision.by_motion;
    return decision;
}

}  // namespace frontend
}  // namespace sfc_core
