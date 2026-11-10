// D5-e：把 detect() 里剩下四处硬编码提成具名配置（R-018）。
//   - 候选上限 5            -> config_.candidate_pool_size（yaml 里本来就有 30，一直没人读）
//   - MAX_SPATIAL_DIST_FOR_LOOP = 8.0 -> loop_max_spatial_dist_m
//   - frames_between < 30   -> min_frames_between_for_spatial_gate
//   - queryWithConsistency(..., 3, 10, 1.0) -> dbow_temporal_window /
//                                              dbow_max_results / dbow_min_distance_m
const fs = require('fs');

// ---------- 1) LoopDetectorConfig 加字段 ----------
const hppPath = 'src/sfc_core/include/sfc_core/loopdetection/loop_detector.hpp';
let hpp = fs.readFileSync(hppPath, 'utf8');

const anchor = `    double frame_dist_min = 0.01;  // 最小帧间距离（米），低于此值不注入里程计尺度`;
if (hpp.indexOf(anchor) < 0) throw new Error('未找到 frame_dist_min 锚点');

const insert = `    // D5-e：DBoW2 查询参数与候选筛选门槛（原来写死在 detect() 里，R-018）
    int dbow_temporal_window = 3;        // queryWithConsistency 的时序窗口
    int dbow_max_results = 10;           // queryWithConsistency 最多返回多少个候选
    double dbow_min_distance_m = 1.0;    // 与当前帧距离小于此值的候选直接丢弃
    double loop_max_spatial_dist_m = 8.0;  // 空间距离上限：超过它且帧间隔够近说明只是"路过"
    int min_frames_between_for_spatial_gate = 30;  // 帧间隔小于此值才启用上面的空间距离闸门

`;
hpp = hpp.replace(anchor, insert + anchor);
fs.writeFileSync(hppPath, hpp);
console.log('[D5-e] LoopDetectorConfig 新增 5 个字段');

// ---------- 2) detect() 用配置 ----------
const cppPath = 'src/sfc_core/src/loopdetection/loop_detector.cpp';
let s = fs.readFileSync(cppPath, 'utf8');

const oldQuery = `        auto dbow_results = place_recog_->queryWithConsistency(
            curr_desc, curr_pose_world.t, recent_matches,
            3, 10, 1.0);`;

const newQuery = `        // D5-e：三个阈值来自配置，不再写死（R-018）。
        auto dbow_results = place_recog_->queryWithConsistency(
            curr_desc, curr_pose_world.t, recent_matches,
            config_.dbow_temporal_window,
            config_.dbow_max_results,
            config_.dbow_min_distance_m);`;

if (s.indexOf(oldQuery) < 0) throw new Error('未找到 queryWithConsistency 调用');
s = s.replace(oldQuery, newQuery);

const oldGate = `            constexpr double MAX_SPATIAL_DIST_FOR_LOOP = 8.0;
            for (size_t i = 0; i < std::min(5ul, dbow_results.size()); ++i) {`;

const newGate = `            // D5-e：候选上限用 candidate_pool_size，空间距离与帧间隔门槛用配置项。
            const double max_spatial_dist = config_.loop_max_spatial_dist_m;
            const int min_frames_gap = config_.min_frames_between_for_spatial_gate;
            const size_t max_bow_cands =
                std::min(static_cast<size_t>(max_bow_candidates), dbow_results.size());
            for (size_t i = 0; i < max_bow_cands; ++i) {`;

if (s.indexOf(oldGate) < 0) throw new Error('未找到 MAX_SPATIAL_DIST_FOR_LOOP 块');
s = s.replace(oldGate, newGate);

const oldCond = `                    if (spatial_dist > MAX_SPATIAL_DIST_FOR_LOOP && frames_between >= 0 && frames_between < 30) {`;
const newCond = `                    if (spatial_dist > max_spatial_dist && frames_between >= 0 &&
                        frames_between < min_frames_gap) {`;
if (s.indexOf(oldCond) < 0) throw new Error('未找到空间距离判定');
s = s.replace(oldCond, newCond);

fs.writeFileSync(cppPath, s);
console.log('[D5-e] detect() 已改用配置项');
