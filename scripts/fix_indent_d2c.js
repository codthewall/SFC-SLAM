// 纯缩进修正：把 D2-c 插入的 separation_m 块内部补齐到 16 空格。
const fs = require('fs');
const p = 'src/sfc_core/src/loopdetection/loop_detector.cpp';
let lines = fs.readFileSync(p, 'utf8').split('\n');

const iSep = lines.findIndex(l => l.includes('double separation_m = frame_dist;'));
if (iSep < 0) throw new Error('未找到 separation_m');

// 紧随其后的 '{' 行
let iOpen = iSep + 1;
while (iOpen < lines.length && lines[iOpen].trim() !== '{') iOpen++;
if (lines[iOpen].trim() !== '{') throw new Error('未找到块起点');

// 找配对的 '}'：用深度计数
let depth = 0;
let iClose = -1;
for (let i = iOpen; i < lines.length; i++) {
  for (const ch of lines[i]) {
    if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) { iClose = i; break; }
    }
  }
  if (iClose >= 0) break;
}
if (iClose < 0) throw new Error('未找到块终点');

for (let i = iOpen; i <= iClose; i++) {
  const t = lines[i].replace(/^\s+/, '');
  lines[i] = '            ' + t;
}

fs.writeFileSync(p, lines.join('\n'));
console.log(`[D2-c] 重排缩进 ${iOpen + 1}..${iClose + 1}`);
