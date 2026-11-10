// D2-c：帧间距离闸门不再用漂移位姿的差值（按行改写，避免多行片段匹配脆弱）。
const fs = require('fs');

const p = 'src/sfc_core/src/loopdetection/loop_detector.cpp';
let lines = fs.readFileSync(p, 'utf8').split('\n');

// 1) 找到 frame_dist 定义行，后面插入 depth-measured separation
const iFrameDist = lines.findIndex(l => l.includes('double frame_dist = delta_t.norm();'));
if (iFrameDist < 0) throw new Error('未找到 frame_dist 定义');

// 2) 找到 t_norm_em > 1e-6 的分支起点，确保插入点在分支内部
const iTNorm = lines.findIndex((l, i) => i > iFrameDist && l.includes('if (t_norm_em > 1e-6) {'));
if (iTNorm < 0) throw new Error('未找到 t_norm_em 分支');

const insert = [
  '        // D2-c：帧间距离闸门改用**深度实测基线**，不再用位姿差值。',
  '        // pose_curr.t - pose_matched.t 是两个 PnP 结果之差，而回环场景里恰恰是它们漂得最厉害；',
  '        // 拿一个已知在漂的量去判断"这两帧隔得够不够远"，闸门本身没有判据。',
  '        // 每对有效对应点给一个平移样本 t = p_train - R_em * p_query，|t| 的中位数就是',
  '        // 两相机之间的真实间隔（RGB-D 有深度，尺度不用猜）；样本 < 3 对时才退回位姿差值。',
  '        double separation_m = frame_dist;',
  '        {',
  '            std::vector<double> baseline_norms;',
  '            baseline_norms.reserve(valid_matches.size());',
  '            const size_t n_sep = std::min(valid_matches.size(),',
  '                                          std::min(pts3d_query_cam_valid.size(),',
  '                                                   pts3d_train_cam_valid.size()));',
  '            const Eigen::Matrix3d R_sep = T_em.block<3, 3>(0, 0);',
  '            for (size_t k = 0; k < n_sep; ++k) {',
  '                const Eigen::Vector3d t_sample =',
  '                    pts3d_train_cam_valid[k] - R_sep * pts3d_query_cam_valid[k];',
  '                if (!t_sample.allFinite()) continue;',
  '                baseline_norms.push_back(t_sample.norm());',
  '            }',
  '            if (baseline_norms.size() >= 3) {',
  '                const size_t mid = baseline_norms.size() / 2;',
  '                std::nth_element(baseline_norms.begin(),',
  '                                 baseline_norms.begin() + mid,',
  '                                 baseline_norms.end());',
  '                separation_m = baseline_norms[mid];',
  '            }',
  '        }',
];

// 插到 if (t_norm_em > 1e-6) { 之后
lines.splice(iTNorm + 1, 0, ...insert);

// 3) 闸门条件与文案改 separation_m
let gateChanged = 0;
for (let i = 0; i < lines.length; i++) {
  const l = lines[i];
  if (l.includes('if (frame_dist < config_.frame_dist_min)')) {
    lines[i] = l.replace('frame_dist', 'separation_m');
    gateChanged++;
  } else if (l.includes('if (frame_dist >= config_.frame_dist_max)')) {
    lines[i] = l.replace('frame_dist', 'separation_m');
    gateChanged++;
  } else if (l.includes('[GeoVerif] EM 拒绝：帧间距离') && l.includes('frame_dist')) {
    lines[i] = l.replace(/frame_dist/g, 'separation_m').replace('帧间距离', '深度实测基线');
    gateChanged++;
  }
}
if (gateChanged < 4) throw new Error('闸门改写不完整: ' + gateChanged);

// 4) 注释里"此检查基于 PnP 漂移轨迹"更新措辞
for (let i = 0; i < lines.length; i++) {
  if (lines[i].includes('此检查基于 PnP 漂移轨迹')) {
    lines[i] = '            // 此检查基于匹配点深度实测基线，不适用于已修正的 ICP/Sim3 路径';
  }
}

fs.writeFileSync(p, lines.join('\n'));
console.log(`[D2-c] separation_m 已插入，闸门改写 ${gateChanged} 处`);
