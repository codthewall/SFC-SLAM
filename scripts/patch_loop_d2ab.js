// D2-a/D2-b 收尾：EM 降级路径的基线不再有"编一个数"的分支。
//
// 现状（本次要替换的区间，行号会在运行前重新定位）：
//   - 两个几乎重复的循环，一个算 |t_sample| 中位数给 separation_m，一个算投影中位数给 baseline；
//   - baseline 取不到时退回 `frame_dist`（`pose_curr.t - pose_matched.t`）—— 那恰恰是漂得最厉害的量；
//   - 缺少"样本不足就直接拒绝"的分支。
//
// 替换为一个循环同时产出两份样本，并在样本不足时拒绝（D2-b）。
const fs = require('fs');

const p = 'src/sfc_core/src/loopdetection/loop_detector.cpp';
const lines = fs.readFileSync(p, 'utf8').split('\n');

const startMark = '        // ★★★ 修复 EM 回退翻译量';
const endMark = '            T_em.block<3,1>(0,3) = Eigen::Vector3d::Zero();';

const startIdx = lines.findIndex((l) => l.startsWith(startMark));
if (startIdx < 0) throw new Error('未找到 EM 基线块起点');
const elseIdx = lines.findIndex((l, i) => i > startIdx && l === endMark);
if (elseIdx < 0) throw new Error('未找到 EM 基线块终点');
// else 分支的闭合大括号紧跟其后
if (lines[elseIdx + 1].trim() !== '}') {
  throw new Error('EM 块结构与预期不符: ' + JSON.stringify(lines[elseIdx + 1]));
}
const endIdx = elseIdx + 1;

const block = `        // D2：RGB-D 有深度，EM 的平移尺度不需要猜。
        //
        // EM 解出的 T_em 把 query 相机系映射到 train 相机系（p_train = R_em * p_query + t_em），
        // 所以每一对有效对应点都给出一个平移样本 t = p_train - R_em * p_query。
        // 把样本投影到 EM 给出的单位方向上取中位数，就是用深度直接测出的基线：
        // 与里程计漂移无关，也不是常数（原实现写死 kEMFallbackTranslationScale = 1.0，
        // 等于往图里塞一条尺度是假的回环边，CODE-RULES R-025 的反例）。
        //
        // D2-c 的帧间距离闸门复用同一批样本：separation_m 取样本模长的中位数，
        // 不再用 pose_curr.t - pose_matched.t —— 回环场景里这两帧的 PnP 结果恰恰漂得最厉害，
        // 拿一个已知在漂的量去判断"这两帧隔得够不够远"，闸门本身就没有判据。
        std::vector<double> baseline_samples;
        double separation_m = 0.0;
        if (t_norm_em > 1e-6) {
            const Eigen::Matrix3d R_em = T_em.block<3, 3>(0, 0);
            const size_t n_pairs = std::min(valid_matches.size(),
                                            std::min(pts3d_query_cam_valid.size(),
                                                     pts3d_train_cam_valid.size()));
            std::vector<double> norm_samples;
            baseline_samples.reserve(n_pairs);
            norm_samples.reserve(n_pairs);
            for (size_t k = 0; k < n_pairs; ++k) {
                const Eigen::Vector3d t_sample =
                    pts3d_train_cam_valid[k] - R_em * pts3d_query_cam_valid[k];
                if (!t_sample.allFinite()) continue;
                norm_samples.push_back(t_sample.norm());
                const double proj = t_sample.dot(t_em_dir);
                if (proj > 1e-3) baseline_samples.push_back(proj);
            }
            if (norm_samples.size() >= 3) {
                const size_t mid = norm_samples.size() / 2;
                std::nth_element(norm_samples.begin(),
                                 norm_samples.begin() + mid,
                                 norm_samples.end());
                separation_m = norm_samples[mid];
            }
        }

        // D2-b：拿不到可靠基线就拒绝这次 EM 回环，而不是编一个数顶上。
        // 早期版本退回漂移的 frame_dist，中间版本退回常数 1.0 m —— 两者都会让一条
        // 尺度和真实几何无关的边进 PGO。与其给 PGO 一条假约束，不如这一次没有约束，
        // 下一个关键帧还会再试（回环检测每次关键帧都会跑）。
        if (baseline_samples.size() < 3 || separation_m <= 1e-3) {
            { if (config_.verbose) std::cerr << "[GeoVerif] EM 拒绝：深度对应点不足以测基线（样本 "
                                            << baseline_samples.size() << " 对）\\n"; }
            { if (config_.verbose) std::cerr << std::string(80, '=') << "\\n"; }
            return false;
        }

        double baseline = 0.0;
        {
            const size_t mid = baseline_samples.size() / 2;
            std::nth_element(baseline_samples.begin(),
                             baseline_samples.begin() + mid,
                             baseline_samples.end());
            baseline = baseline_samples[mid];
        }
        if (!std::isfinite(baseline) || baseline <= 1e-3) {
            { if (config_.verbose) std::cerr << "[GeoVerif] EM 拒绝：基线非有限或过小（"
                                            << baseline << " m）\\n"; }
            { if (config_.verbose) std::cerr << std::string(80, '=') << "\\n"; }
            return false;
        }
        T_em.block<3, 1>(0, 3) = baseline * t_em_dir;
        if (config_.verbose) {
            std::cerr << "[GeoVerif] EM 基线（深度实测，中位数）：" << baseline
                      << " m，帧间基线 " << separation_m
                      << " m（样本 " << baseline_samples.size() << " 对）" << std::endl;
        }
        // 到这里 T_em 的平移已有真实尺度；t_norm_em == 0 的分支（recoverPose 未给出平移）
        // 已被上面的样本检查挡掉，不再需要"补零"的兜底。
`.split('\n');

lines.splice(startIdx, endIdx - startIdx + 1, ...block.slice(0, block.length - 1));
fs.writeFileSync(p, lines.join('\n'));
console.log('[D2-a/b] EM 基线块已重写，行数', lines.length);
