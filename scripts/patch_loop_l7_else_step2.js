#!/usr/bin/env node
// 一次性补丁第二步：上一步的括号配平把 `} else if (...) {` 这一行本身判成了
// "已闭合"（该行同时含 } 和 {），结果只替换了首行，分支体残留下来。
// 这里按行号删除残留的分支体（从 "ICP 内点为0" 注释行到最后那个孤立的 "}"）。
const fs = require('fs');
const path = process.argv[2] || 'src/sfc_core/src/loopdetection/loop_detector.cpp';

const lines = fs.readFileSync(path, 'utf8').split('\n');

let start = -1;
for (let i = 0; i < lines.length; ++i) {
    if (lines[i].includes('尝试 ICP 旋转+里程计尺度回退')) { start = i; break; }
}
if (start < 0) {
    console.error('未找到残留分支体起点');
    process.exit(1);
}

// 从起点向下配平，找到被删分支的收尾 `}`（残留体本身是花括号平衡的）
let depth = 0, end = -1, started = false;
for (let i = start; i < lines.length; ++i) {
    for (const ch of lines[i]) {
        if (ch === '{') { depth++; started = true; }
        else if (ch === '}') { depth--; }
    }
    if (started && depth === 0) { end = i; break; }
}
if (end < 0) { console.error('括号配平失败'); process.exit(1); }

// 残留体后面还跟着原 else-if 的收尾 `}`（第 end+1 行），一并删掉
let extra = 0;
if (lines[end + 1] !== undefined && lines[end + 1].trim() === '}') { extra = 1; }

const out = lines.slice(0, start).concat(lines.slice(end + 1 + extra));
fs.writeFileSync(path, out.join('\n'), 'utf8');
console.log(`已删除残留 ${path} 第 ${start + 1}-${end + 1 + extra} 行（共 ${end + 1 + extra - start} 行）`);
