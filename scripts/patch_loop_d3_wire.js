// D3-c / D3-d：把剩下的重复键清掉，并把 geometric_verifier.yaml 真正接到 LoopDetector。
//
// 1) loadAllConfigs()：geometric_verifier.yaml 读进 AllConfigs 之后就没人用了 ——
//    这里补上第二跳：把整份 GeometricVerifierConfig 放进 loop_detector。
// 2) LoopDetectorConfig 里 ransac_iterations / ransac_threshold_px / max_icp_inlier_ratio
//    三个字段与 GeometricVerifierConfig 同名同义，是同一批参数的第二份声明（R-023）。
//    删掉它们，loader 与 yaml 的对应条目一起删。
//    —— 注意 ransac_threshold_px=8.0 与 geometric_verifier.yaml 的 threshold_px=4.0
//       原本就不一致，删掉后以 geometric_verifier.yaml 的 4.0 为准（这才是几何验证器
//       真正使用的值：构造函数以前把它覆盖成 8.0，等于 yaml 里那份配了不生效）。
const fs = require('fs');

// ---------- 1) loader：补第二跳 ----------
const loaderPath = 'src/sfc_core/src/config/config_loader.cpp';
let loader = fs.readFileSync(loaderPath, 'utf8');

const anchor = `    all_cfg.loop_detector = loadLoopDetectorConfig(
        makePath(config_dir, "loop_detector.yaml"));
`;

const anchorNew = `    all_cfg.loop_detector = loadLoopDetectorConfig(
        makePath(config_dir, "loop_detector.yaml"));
    // D3-d：geometric_verifier.yaml 以前只读进 AllConfigs 就断了，没有任何消费者，
    // 几何验证器实际用的是 LoopDetector 构造函数里的字面量。这里把整份配置塞进
    // loop_detector，由构造函数原样交给 GeometricVerifier —— 一条链，不再抄字段。
    all_cfg.loop_detector.geometric_verifier = all_cfg.geometric_verifier;
`;

if (loader.indexOf(anchor) < 0) throw new Error('未找到 loadLoopDetectorConfig 调用点');
loader = loader.replace(anchor, anchorNew);

// 删除三个重复键的读取（它们已从结构体移除，留着会编译失败）
const dupReads = [
  `    cfg.ransac_iterations = getInt(fs, "ransac_iterations", cfg.ransac_iterations);\n`,
  `    cfg.ransac_threshold_px = getDouble(fs, "ransac_threshold_px", cfg.ransac_threshold_px);\n`,
  `    cfg.max_icp_inlier_ratio = getDouble(fs, "max_icp_inlier_ratio", cfg.max_icp_inlier_ratio);\n`,
];
for (const d of dupReads) {
  if (loader.indexOf(d) < 0) throw new Error('未找到重复键读取: ' + d.trim());
  loader = loader.replace(d, '');
}
fs.writeFileSync(loaderPath, loader);
console.log('[D3] config_loader：注入 geometric_verifier，删除三个重复键读取');

// ---------- 2) 头文件：删三个重复字段 ----------
const hppPath = 'src/sfc_core/include/sfc_core/loopdetection/loop_detector.hpp';
let hpp = fs.readFileSync(hppPath, 'utf8');

const oldFields = `    // 几何校验
    int ransac_iterations = 200;
    double ransac_threshold_px = 8.0;  // RANSAC 内点阈值（像素）
    int min_geometric_inliers = 6;      // 几何校验最少内点数（从 8 降低到 6，提高回环检出率）

    // ICP 验证参数
    double max_icp_inlier_ratio = 0.70; // ★ ICP 内点率上限（>70% 说明退化，拒绝）
`;

const newFields = `    // 几何校验
    // ransac_iterations / ransac_threshold_px / max_icp_inlier_ratio 原本也在这里，
    // 与 GeometricVerifierConfig 的同名字段是同一批参数的两份声明（R-023）：
    // 构造函数会把它们覆盖进几何验证器，于是 geometric_verifier.yaml 里配的值永远不生效。
    // 现在删掉，只留几何验证器那一份（见下面的 geometric_verifier 成员）。
    int min_geometric_inliers = 6;      // detect() 单候选路径的最少内点数（不是几何验证器内部阈值）
`;

if (hpp.indexOf(oldFields) < 0) throw new Error('未找到三个重复字段声明');
hpp = hpp.replace(oldFields, newFields);
fs.writeFileSync(hppPath, hpp);
console.log('[D3] loop_detector.hpp：删除三个重复字段声明');

// ---------- 3) yaml：删掉 loop_detector.yaml 里的重复条目 ----------
const yamlPath = 'config/loop_detector.yaml';
let yaml = fs.readFileSync(yamlPath, 'utf8');

const oldYaml = `# RANSAC 迭代次数。
# 与 GeometricVerifier 中的 ransac_iterations 协同工作。
# 建议值：200-500。
ransac_iterations: 200

# RANSAC 内点阈值（像素）。
# 注意：此参数在 Tracking.cpp 中运行时覆盖为 8.0。
# 与 GeometricVerifier 中的 threshold_px (4.0) 不同！
# 建议值：6.0-10.0。
ransac_threshold_px: 8.0
`;

const newYaml = `# ⚠ RANSAC 迭代次数与内点阈值（ransac_iterations / threshold_px）已移到
#   config/geometric_verifier.yaml —— 那是几何验证器读取的唯一来源。
#   本文件曾有同名条目，构造函数会覆盖进去，导致 geometric_verifier.yaml 配了不生效。
`;

if (yaml.indexOf(oldYaml) < 0) throw new Error('未找到 loop_detector.yaml 的 ransac 条目');
yaml = yaml.replace(oldYaml, newYaml);

const oldMaxIcp = `# ICP 内点率上限（> 此值说明退化，拒绝 ICP）。
# 与 GeometricVerifier 中的 max_icp_inlier_ratio 保持一致。
# 建议值：0.65-0.75。
max_icp_inlier_ratio: 0.70
`;

const newMaxIcp = `# ⚠ max_icp_inlier_ratio 同上，唯一来源是 config/geometric_verifier.yaml。
`;

if (yaml.indexOf(oldMaxIcp) < 0) throw new Error('未找到 loop_detector.yaml 的 max_icp_inlier_ratio');
yaml = yaml.replace(oldMaxIcp, newMaxIcp);
fs.writeFileSync(yamlPath, yaml);
console.log('[D3] loop_detector.yaml：删除三个重复条目');
