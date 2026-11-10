// L9 + D2：EM 降级路径的真实基线（把 kEMFallbackTranslationScale=1.0 换成深度实测）
// 死代码 correctLoopWithScale() 想做的事（给 EM 无尺度平移补一个真实尺度）方向是对的，
// 但它取的是两帧**漂移后**位姿的距离 —— 那正是本段注释反复警告的不可信来源。
// 正确来源是手上已有的 3D 对应点：T_em 把 query 相机系映射到 train 相机系，
// 于是 t = p_train - R_em * p_query 每一对都给出一个平移样本，投影到 t_dir 上取中位数，
// 就是由深度直接测出来的基线，不掺任何里程计漂移。
const fs = require('fs');

const path = 'src/sfc_core/src/loopdetection/loop_detector.cpp';
let src = fs.readFileSync(path, 'utf8');

const oldBlock = `        // ★★★ 修复 EM 回退翻译量：不再使用漂移的 odometry 估计 ★★★
        // 原代码用 pose_curr.t - pose_matched.t 作为平移量，这是 PnP 累积漂移后的值。
        // 例如：fr1_desk2 室内真实运动 1-3m，但 PnP 漂移后 odometry 估计可能达到 5-8m。
        // 直接用漂移的 odometry 距离作为 EM 平移量，会让 PGO 的 Sim3 约束完全错误。
        //
        // 修复：保持 Essential Matrix 归一化翻译方向，不使用 odometry 尺度。
        // RGB-D 有深度约束，不需要依赖 EM 的尺度估计。
        // Sim3 的尺度信息来自 ICP 路径（有真实深度尺度）。
        // 设为 1.0m 作为合理的室内相机间距估计。
        constexpr double kEMFallbackTranslationScale = 1.0;
        Eigen::Vector3d delta_t = pose_curr.t - pose_matched.t;
        double frame_dist = delta_t.norm();
        if (t_norm_em > 1e-6) {
            { if (config_.verbose) std::cerr << "[GeoVerif] ★ EM 平移使用固定尺度(不依赖odometry): |t|=" << kEMFallbackTranslationScale << " m" << std::endl; }
            T_em.block<3,1>(0,3) = kEMFallbackTranslationScale * t_em_dir;
        } else {
            T_em.block<3,1>(0,3) = Eigen::Vector3d::Zero();
        }
`;

const newBlock = `        // ★★★ 修复 EM 回退翻译量：不再使用漂移的 odometry 估计 ★★★
        // 原代码用 pose_curr.t - pose_matched.t 作为平移量，这是 PnP 累积漂移后的值。
        // 例如：fr1_desk2 室内真实运动 1-3m，但 PnP 漂移后 odometry 估计可能达到 5-8m。
        // 中间版本改成写死 kEMFallbackTranslationScale = 1.0 —— 那是"拿不到尺度就用常数顶替"
        // （CODE-RULES R-025 的反例）：方向对、长度永远是 1.0m，回环边带着这个假尺度进 PGO。
        //
        // L9 + D2：RGB-D 有深度，尺度不需要猜。EM 解出的 T_em 把 query 相机系映射到
        // train 相机系（p_train = R_em * p_query + t_em），所以每一对有效对应点都给一个
        // 平移样本 t = p_train - R_em * p_query。把样本投影到 EM 给出的单位方向上取中位数，
        // 得到的就是由深度直接测出的基线 —— 与里程计漂移无关，也不是常数。
        Eigen::Vector3d delta_t = pose_curr.t - pose_matched.t;
        double frame_dist = delta_t.norm();
        if (t_norm_em > 1e-6) {
            const Eigen::Matrix3d R_em = T_em.block<3,3>(0,0);
            std::vector<double> baseline_samples;
            baseline_samples.reserve(valid_matches.size());
            const size_t n_pairs = std::min(valid_matches.size(),
                                            std::min(pts3d_query_cam_valid.size(),
                                                     pts3d_train_cam_valid.size()));
            for (size_t k = 0; k < n_pairs; ++k) {
                const Eigen::Vector3d t_sample =
                    pts3d_train_cam_valid[k] - R_em * pts3d_query_cam_valid[k];
                if (!t_sample.allFinite()) continue;
                const double proj = t_sample.dot(t_em_dir);
                if (proj > 1e-3) baseline_samples.push_back(proj);
            }

            double baseline = 0.0;
            if (!baseline_samples.empty()) {
                const size_t mid = baseline_samples.size() / 2;
                std::nth_element(baseline_samples.begin(),
                                 baseline_samples.begin() + mid,
                                 baseline_samples.end());
                baseline = baseline_samples[mid];
            }
            if (!std::isfinite(baseline) || baseline <= 1e-3) {
                // 深度对应点全被剔除时才退回帧间距离：它至少是"这两帧相隔多远"的量级，
                // 仍不是常数，也不会把 1.0m 硬塞进图里。
                baseline = frame_dist;
                if (config_.verbose) {
                    std::cerr << "[GeoVerif] EM 基线：深度样本不足，退回帧间距离 "
                              << baseline << " m" << std::endl;
                }
            } else if (config_.verbose) {
                std::cerr << "[GeoVerif] EM 基线（深度实测，中位数）：" << baseline
                          << " m（样本 " << baseline_samples.size() << " 对）" << std::endl;
            }
            T_em.block<3,1>(0,3) = baseline * t_em_dir;
        } else {
            T_em.block<3,1>(0,3) = Eigen::Vector3d::Zero();
        }
`;

if (src.indexOf(oldBlock) < 0) throw new Error('未找到 EM 基线旧代码块');
if (src.indexOf(oldBlock, src.indexOf(oldBlock) + 1) >= 0) throw new Error('EM 基线旧代码块重复');
src = src.replace(oldBlock, newBlock);

fs.writeFileSync(path, src);
console.log('[L9/D2] EM 降级路径已改为深度实测基线');
