// D5-e：把新增的 5 个配置项补进 yaml 与 loader（否则字段只有结构体默认值，yaml 里没法调）。
const fs = require('fs');

const yamlPath = 'config/loop_detector.yaml';
let yaml = fs.readFileSync(yamlPath, 'utf8');

const anchor = `# 最大帧间距离（米），高于此值不注入里程计尺度（避免大尺度漂移）。
# 室内场景：10-20m；户外：50-100m。
frame_dist_max: 50.0
`;
if (yaml.indexOf(anchor) < 0) throw new Error('未找到 frame_dist_max 锚点');

const block = `
# ================================================================
# 八点五、DBoW2 查询与候选空间闸门（D5-e）
# ================================================================
#
# 这四项原来写死在 LoopDetector::detect() 里（R-018）：
#   queryWithConsistency(..., 3, 10, 1.0) 与 MAX_SPATIAL_DIST_FOR_LOOP = 8.0、frames_between < 30
#
# dbow_temporal_window：位置识别的时序窗口（候选必须在这几帧的邻居里也有支持）。
dbow_temporal_window: 3
#
# dbow_max_results：queryWithConsistency 最多返回多少候选（再多也没意义，几何验证才是瓶颈）。
dbow_max_results: 10
#
# dbow_min_distance_m：与当前帧空间距离小于此值的候选直接丢弃 —— 那是"还没走开"，不是回环。
dbow_min_distance_m: 1.0
#
# loop_max_spatial_dist_m：候选与当前帧空间距离超过它、**且**帧间隔小于下面那个值，
# 说明只是"路过同一个地方"而不是真的绕回来了，丢弃。
loop_max_spatial_dist_m: 8.0
#
# min_frames_between_for_spatial_gate：上面那道空间闸门只在帧间隔小于此值时启用。
# （帧间隔本来就大时，空间距离远是正常的，不该被砍。）
min_frames_between_for_spatial_gate: 30
`;

yaml = yaml.replace(anchor, anchor + block);
fs.writeFileSync(yamlPath, yaml);
console.log('[D5-e] yaml 已补 5 项');

// loader
const loaderPath = 'src/sfc_core/src/config/config_loader.cpp';
let loader = fs.readFileSync(loaderPath, 'utf8');

const lAnchor = `    cfg.frame_dist_min = getDouble(fs, "frame_dist_min", cfg.frame_dist_min);
    cfg.frame_dist_max = getDouble(fs, "frame_dist_max", cfg.frame_dist_max);
`;
if (loader.indexOf(lAnchor) < 0) throw new Error('未找到 loader frame_dist 块');

const lBlock = lAnchor + `
    // D5-e：DBoW2 查询与候选空间闸门（原来写死在 detect() 里）
    cfg.dbow_temporal_window = getInt(fs, "dbow_temporal_window", cfg.dbow_temporal_window);
    cfg.dbow_max_results = getInt(fs, "dbow_max_results", cfg.dbow_max_results);
    cfg.dbow_min_distance_m = getDouble(fs, "dbow_min_distance_m", cfg.dbow_min_distance_m);
    cfg.loop_max_spatial_dist_m =
        getDouble(fs, "loop_max_spatial_dist_m", cfg.loop_max_spatial_dist_m);
    cfg.min_frames_between_for_spatial_gate = getInt(
        fs, "min_frames_between_for_spatial_gate", cfg.min_frames_between_for_spatial_gate);
`;

loader = loader.replace(lAnchor, lBlock);
fs.writeFileSync(loaderPath, loader);
console.log('[D5-e] loader 已补 5 项读取');
