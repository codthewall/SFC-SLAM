// D3-a / D3-c / D3-d：几何验证阈值只留一个来源。
//
// 现状（L5 之后）：LoopDetectorConfig 里有一组 geo_* 字段，构造函数再把它们逐个抄进
// GeometricVerifierConfig —— 同一批参数存在两份声明（R-023：单一真值来源）。
// config/geometric_verifier.yaml 也是同一批参数的第三份拷贝，只是 loader 读进
// AllConfigs::geometric_verifier 后没人往下传。
//
// 改成：LoopDetectorConfig 直接持有一个 GeometricVerifierConfig 成员，
// 由 loadAllConfigs() 从 geometric_verifier.yaml 灌进来（yaml → AllConfigs →
// LoopDetectorConfig → LoopDetector），四跳一条链，中间不再抄字段。
const fs = require('fs');

const hppPath = 'src/sfc_core/include/sfc_core/loopdetection/loop_detector.hpp';
let hpp = fs.readFileSync(hppPath, 'utf8');

const oldBlock = `    // ★★★ 几何验证阈值（L5：原来写死在 loop_detector.cpp 构造函数里）★★★
    // config/geometric_verifier.yaml 里一直有一份，但没有任何消费者 ——
    // 改配置不生效、调参只能改源码重编译。现在由 loadAllConfigs() 注入。
    // 默认值等于原写死值，保证"不配也能跑"。
    double geo_min_inlier_ratio = 0.15;        // Essential Matrix 最少内点率
    int    geo_min_inliers = 2;                // Essential Matrix 最少内点数
    int    geo_min_icp_inliers = 2;            // ICP 最少内点数
    double geo_min_icp_inlier_ratio = 0.10;    // ICP 最少内点率
    double geo_icp_reproj_thresh_px = 35.0;    // ICP 重投影误差阈值（像素）
    double geo_min_depth = 0.1;                // 参与几何验证的最小深度（米）
    double geo_max_depth = 12.0;               // 参与几何验证的最大深度（米）
    int    geo_min_essential_inliers = 3;      // Essential 降级路径最少内点
    double geo_min_essential_inlier_ratio = 0.40;
`;

const newBlock = `    // ★★★ 几何验证参数（D3）★★★
    // 直接持有 GeometricVerifierConfig，而不是把它的字段在本结构里再抄一遍。
    // 链路：config/geometric_verifier.yaml → loadGeometricVerifierConfig()
    //       → AllConfigs::geometric_verifier → AllConfigs::loop_detector.geometric_verifier
    //       → LoopDetector 构造函数 → GeometricVerifier。
    // 原实现这条链在第二跳就断了（yaml 读进 AllConfigs 后无人使用），构造函数里
    // 另写一份字面量 —— 改配置不生效、调参只能改源码重编译。
    GeometricVerifierConfig geometric_verifier;
`;

if (hpp.indexOf(oldBlock) < 0) throw new Error('未找到 geo_* 字段块');
hpp = hpp.replace(oldBlock, newBlock);
fs.writeFileSync(hppPath, hpp);
console.log('[D3] LoopDetectorConfig：geo_* 字段 → GeometricVerifierConfig 成员');

// ---- 构造函数不再逐字段搬运 ----
const cppPath = 'src/sfc_core/src/loopdetection/loop_detector.cpp';
let src = fs.readFileSync(cppPath, 'utf8');

const oldCtor = `    GeometricVerifierConfig geo_config;
    geo_config.ransac_iterations = config_.ransac_iterations;
    geo_config.threshold_px = config_.ransac_threshold_px;
    // L5：这里原来写死了 0.15 / 2 / 2 / 0.10 / 35.0 / 0.1 / 12.0，而
    // config/geometric_verifier.yaml 里配好的一组参数从来没有人读 —— 改配置不生效，
    // 调参只能改源码重编译。现在全部改由 LoopDetectorConfig 的 geo_* 字段注入，
    // 默认值仍在头文件里给一份（见 loop_detector.hpp 的 GeometricVerifierConfig 默认块），
    // 保持"不配也能跑"的行为。
    geo_config.min_inlier_ratio = config_.geo_min_inlier_ratio;
    geo_config.min_inliers = config_.geo_min_inliers;
    geo_config.min_icp_inliers = config_.geo_min_icp_inliers;
    geo_config.min_icp_inlier_ratio = config_.geo_min_icp_inlier_ratio;
    geo_config.icp_reproj_thresh_px = config_.geo_icp_reproj_thresh_px;
    geo_config.max_icp_inlier_ratio = config_.max_icp_inlier_ratio;
    geo_config.min_depth = config_.geo_min_depth;
    geo_config.max_depth = config_.geo_max_depth;
    geo_config.min_essential_inliers = config_.geo_min_essential_inliers;
    geo_config.min_essential_inlier_ratio = config_.geo_min_essential_inlier_ratio;
    geo_config.verbose = config_.verbose;
    geometric_verifier_ = std::make_unique<GeometricVerifier>(geo_config);
`;

const newCtor = `    // D3：几何验证参数整份来自 LoopDetectorConfig::geometric_verifier，
    // 这个成员由 loadAllConfigs() 从 config/geometric_verifier.yaml 灌入。
    // 原来这里有一串 "geo_config.xxx = config_.geo_xxx" 的搬运 —— 那是同一批参数
    // 的第二份声明（R-023），改 yaml 不生效、改代码两处要同步。
    GeometricVerifierConfig geo_config = config_.geometric_verifier;
    // verbose 只挂在 LoopDetectorConfig 上（回环层统一开关），几何验证跟随它。
    geo_config.verbose = config_.verbose;
    geometric_verifier_ = std::make_unique<GeometricVerifier>(geo_config);
`;

if (src.indexOf(oldCtor) < 0) throw new Error('未找到构造函数里的 geo_config 搬运块');
src = src.replace(oldCtor, newCtor);
fs.writeFileSync(cppPath, src);
console.log('[D3] LoopDetector 构造函数：改为整份取 config_.geometric_verifier');
