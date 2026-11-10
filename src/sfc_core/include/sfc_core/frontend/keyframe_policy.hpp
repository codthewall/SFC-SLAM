// ================================================================
// 关键帧插入策略（纯逻辑，无 OpenCV / Eigen 依赖）
// ================================================================
//
// 为什么要单独抽出来：
//   原先 needNewKeyFrame() 的判断直接写在 Tracking::track() 的调用链里，
//   六个条件（c1a/c1b/c1c/c2/c1d/c1e）用 || 连在一起，只要一个成立就插帧。
//   其中 c1b 是"距上一关键帧 >= 10 帧"、c1c 是"内点数 < 参考匹配数 25%"
//   且两者都没有帧间隔下限 —— 结果是几乎每帧都插关键帧
//   （freiburg1_desk2 实测 640 帧出 562 个关键帧）。
//   关键帧一多，局部地图窗口（固定 20）里全是时间上相邻的帧，
//   视差接近零，三角化退化，位姿就再也拉不回来。
//
//   把判定抽成纯函数后：
//     1. 有唯一的事实来源（本文件），不再散落在 3900 行的 tracking.cpp 里；
//     2. 能脱离相机数据单测（test/test_keyframe_policy.cpp）；
//     3. 帧间隔下限成为策略的一部分，而不是靠调用方自觉。
//
// 判定顺序（短路）：
//   强制插入（间隔上限） > 兜底插入（长期内点极少） > 质量/运动触发（须满足最小间隔）
//
// ================================================================
#ifndef SFC_CORE_FRONTEND_KEYFRAME_POLICY_HPP
#define SFC_CORE_FRONTEND_KEYFRAME_POLICY_HPP

namespace sfc_core {
namespace frontend {

// 关键帧策略参数。
// 与 TrackingConfig 中的同名字段一一对应，由 tracking.yaml 提供。
struct KeyframePolicyParams {
    // 两个关键帧之间的最小帧间隔（硬下限）。
    // 0 或 1 表示不做限制（不建议：会让关键帧退化成逐帧插入）。
    int min_gap_frames = 8;

    // 强制插入间隔（帧）。距上一关键帧超过此值无条件插入，防止静止时地图停更。
    // <= 0 表示关闭强制插入。
    int forced_interval = 30;

    // 内点绝对下限：内点数高于此值时，不因为"相对内点率下降"而插帧。
    int min_tracked_inliers = 120;

    // 相对内点率：内点数 < ref_matches * weak_tracked_ratio 视为跟踪质量下降。
    double weak_tracked_ratio = 0.30;

    // 运动阈值：相对上一关键帧的位移 / 转角超过阈值时插帧。
    double keyframe_translation = 0.15;   // 米
    double keyframe_rotation = 0.05;      // 弧度

    // 兜底：内点数长期低于 rescue_min_inliers 时，即使没到强制间隔也插一帧，
    // 用新关键帧重建一批可用的地图点。要求已间隔 rescue_max_gap 帧。
    int rescue_min_inliers = 40;
    int rescue_max_gap = 60;
};

// 一次判定的输入证据。全部由 Tracking 在调用点算出，本模块不碰任何框架类型。
struct KeyframeEvidence {
    int frame_gap = 0;              // 当前帧 id - 上一关键帧 id
    int inliers = 0;                // 当前帧 poseOptimization 的内点数
    int ref_matches = 0;            // 参考关键帧中仍然活着的地图点数量
    bool velocity_valid = false;    // 运动模型是否可信（否则不看运动阈值）
    double translation_since_kf = 0.0;  // 相对上一关键帧的平移（米）
    double rotation_since_kf = 0.0;     // 相对上一关键帧的旋转（弧度）
};

// 判定结果。保留分支标记，便于诊断日志说明"为什么插了这一帧"。
struct KeyframeDecision {
    bool need_keyframe = false;
    bool forced = false;            // 由 forced_interval 触发
    bool rescue = false;            // 由 rescue_* 触发
    bool by_tracking_quality = false;  // 由内点数下降触发
    bool by_motion = false;         // 由位移/转角触发
};

// 根据参数与证据决定是否插入关键帧。
// 纯函数：同样的输入永远得到同样的输出，不读写任何全局状态。
KeyframeDecision decideKeyframe(const KeyframePolicyParams& params,
                                const KeyframeEvidence& evidence);

}  // namespace frontend
}  // namespace sfc_core

#endif  // SFC_CORE_FRONTEND_KEYFRAME_POLICY_HPP
