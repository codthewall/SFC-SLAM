// D4-d：回环检测/PGO 的节流开关。
//
// 现状：Tracking::detectAndCorrectLoop() 每个关键帧都会跑一次检测 + 一次 PGO，
// 唯一的节流是"同一个 KF 不重复进来"。而 CovisManager 那条路径有自己的
// loop_detection_interval（config/covis_manager.yaml 里是 9999，等于关闭），
// 两条路径的节流语义完全不一致，谁也没法用一个值描述"多久检测一次"。
//
// 这里给 Tracking 路径加一个明确的最小关键帧间隔：默认 1 = 每个关键帧都检测
// （与当前行为完全一致，不改变默认运行结果），需要降 CPU 时把它调大。
const fs = require('fs');

function patch(path, pairs) {
  let s = fs.readFileSync(path, 'utf8');
  for (const [from, to, label] of pairs) {
    if (s.indexOf(from) < 0) throw new Error(path + ' :: 未找到片段 ' + label);
    s = s.replace(from, to);
  }
  fs.writeFileSync(path, s);
  console.log('[D4-d] patched', path);
}

// 1) LoopDetectorConfig 增加字段
patch('src/sfc_core/include/sfc_core/loopdetection/loop_detector.hpp', [
  [
`    int min_frames_between_for_spatial_gate = 30;`,
`    int min_frames_between_for_spatial_gate = 30;

    // D4-d：回环检测/PGO 的最小关键帧间隔。
    // 1 = 每个关键帧都检测（与改动前的实际行为一致）；调大可以显著降 CPU，
    // 代价是同一个回环要等更久才被发现。
    // 注意这与 CovisManager 的 loop_detection_interval（按帧计数）不是同一个量，
    // 两者分别作用于两条不同的回环路径。
    int loop_check_interval_kf = 1;`,
    'config field',
  ],
]);

// 2) Tracking 成员：保存间隔与上次检测的 KF id（已有 mnLastLoopCheckKFId_，复用）
patch('src/sfc_core/include/sfc_core/frontend/tracking.hpp', [
  [
`    // 上一次做回环检测的关键帧 ID；用于按 FORCED 间隔节流（回环检测比建关键帧更贵）
    int mnLastLoopCheckKFId_ = -1;`,
`    // 上一次做回环检测的关键帧 ID；用于按 FORCED 间隔节流（回环检测比建关键帧更贵）
    int mnLastLoopCheckKFId_ = -1;
    // D4-d：两次回环检测之间至少间隔多少个关键帧（来自 LoopDetectorConfig::loop_check_interval_kf）
    int mnLoopCheckIntervalKF_ = 1;`,
    'tracking member',
  ],
]);

// 3) initLoopDetection 里写入
patch('src/sfc_core/src/frontend/tracking.cpp', [
  [
`    loopdetection::LoopDetectorConfig cfg = loop_cfg;
    cfg.camera_K = config_.K;
    cfg.camera_D = config_.D;`,
`    loopdetection::LoopDetectorConfig cfg = loop_cfg;
    cfg.camera_K = config_.K;
    cfg.camera_D = config_.D;
    // D4-d：检测节流来自配置，默认 1（每个关键帧）
    mnLoopCheckIntervalKF_ = std::max(1, cfg.loop_check_interval_kf);`,
    'init store interval',
  ],
  [
`    if (mnLastLoopCheckKFId_ == kf.frame_id) return false;
    mnLastLoopCheckKFId_ = kf.frame_id;`,
`    if (mnLastLoopCheckKFId_ == kf.frame_id) return false;
    // D4-d：按关键帧间隔节流（默认 1 = 不节流，与改动前行为一致）。
    // 回环检测比建关键帧贵得多：一次 detect() 要做 BoW 查询 + 几何验证 + Sim3/ICP，
    // 而 PGO 是全局优化。此前唯一的闸门是"同一 KF 不重复进来"，等于每个关键帧都全跑一遍。
    if (mnLoopCheckIntervalKF_ > 1 && mnLastLoopCheckKFId_ >= 0 &&
        kf.frame_id - mnLastLoopCheckKFId_ < mnLoopCheckIntervalKF_) {
        return false;
    }
    mnLastLoopCheckKFId_ = kf.frame_id;`,
    'throttle',
  ],
]);

// 4) yaml + loader
patch('config/loop_detector.yaml', [
  [
`min_frames_between_for_spatial_gate: 30`,
`min_frames_between_for_spatial_gate: 30

# ----------------------------------------------------------------
# 八点六、回环检测节流（D4-d）
# ----------------------------------------------------------------
# 每多少个关键帧才做一次回环检测（连带一次 PGO）。
# 1 = 每个关键帧都检测（改动前的实际行为，默认）。
# 调大可以显著降 CPU：一次 detect() 要做 BoW 查询 + 几何验证 + Sim3/ICP，
# PGO 还是全局优化。代价是同一个回环要等更久才会被发现。
loop_check_interval_kf: 1`,
    'yaml key',
  ],
]);

patch('src/sfc_core/src/config/config_loader.cpp', [
  [
`    cfg.min_frames_between_for_spatial_gate = getInt(
        fs, "min_frames_between_for_spatial_gate", cfg.min_frames_between_for_spatial_gate);`,
`    cfg.min_frames_between_for_spatial_gate = getInt(
        fs, "min_frames_between_for_spatial_gate", cfg.min_frames_between_for_spatial_gate);
    // D4-d：回环检测节流（关键帧计）
    cfg.loop_check_interval_kf =
        getInt(fs, "loop_check_interval_kf", cfg.loop_check_interval_kf);`,
    'loader key',
  ],
]);

console.log('[D4-d] done');
