// D2-a/b 插入块的缩进归一：新块位于 if (em_inliers > 0) { 内部，应当是 12 空格起。
// 按起止标记之间的非空行整体右移 4 格，其余行不动。
const fs = require('fs');

const p = 'src/sfc_core/src/loopdetection/loop_detector.cpp';
const lines = fs.readFileSync(p, 'utf8').split('\n');

const startMark = '        // D2：RGB-D 有深度，EM 的平移尺度不需要猜。';
const endMark = '        // 已被上面的样本检查挡掉，不再需要"补零"的兜底。';

const s = lines.indexOf(startMark);
if (s < 0) throw new Error('未找到 D2 块起点');
const e = lines.indexOf(endMark);
if (e < 0) throw new Error('未找到 D2 块终点');

for (let i = s; i <= e; ++i) {
  if (lines[i].trim() === '') continue;
  lines[i] = '    ' + lines[i];
}

fs.writeFileSync(p, lines.join('\n'));
console.log('[diag] 已把 D2 块缩进右移 4 格，行区间', s + 1, '-', e + 1);
