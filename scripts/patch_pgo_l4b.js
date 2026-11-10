// L4（重做）：addLoopEdge 返回 bool —— 按函数体范围改写，不依赖零散片段匹配
const fs = require('fs');

const path = 'src/sfc_core/src/loopdetection/pose_graph_optimizer.cpp';
let lines = fs.readFileSync(path, 'utf8').split('\n');

const startIdx = lines.findIndex(l => l.startsWith('void PoseGraphOptimizer::addLoopEdge('));
if (startIdx < 0) throw new Error('未找到 addLoopEdge 定义');
const endIdx = lines.findIndex((l, i) => i > startIdx && l.startsWith('int PoseGraphOptimizer::optimize()'));
if (endIdx < 0) throw new Error('未找到 optimize() 定义，函数边界不明');

// 函数体最后一行应当是 '}'；其前一行是 optimizerNeedsRebuild_ = true;
let bodyEnd = endIdx - 1;
while (bodyEnd > startIdx && lines[bodyEnd].trim() === '') bodyEnd--;
if (lines[bodyEnd].trim() !== '}') {
    throw new Error(`函数结束行不是 '}'，实际是：${JSON.stringify(lines[bodyEnd])}`);
}

lines[startIdx] = lines[startIdx].replace(
    'void PoseGraphOptimizer::addLoopEdge(',
    'bool PoseGraphOptimizer::addLoopEdge(');

let converted = 0;
for (let i = startIdx; i <= bodyEnd; ++i) {
    if (lines[i].trim() === 'return;') {
        lines[i] = lines[i].replace('return;', 'return false;');
        ++converted;
    }
}

// 成功出口：在函数体闭合 '}' 之前插一条 return true;
const lastRebuild = (() => {
    for (let i = bodyEnd - 1; i > startIdx; --i) {
        if (lines[i].includes('optimizerNeedsRebuild_ = true;')) return i;
    }
    throw new Error('未找到成功出口前的 optimizerNeedsRebuild_ = true;');
})();
lines.splice(lastRebuild + 1, 0, '    // L4：边真的进了图 —— 调用方据此决定是否计一次有效回环。',
                              '    return true;');

fs.writeFileSync(path, lines.join('\n'));
console.log(`[L4] addLoopEdge 行 ${startIdx + 1}-${bodyEnd + 1}：${converted} 处 return; -> return false;，尾部补 return true;`);
