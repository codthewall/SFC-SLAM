// L2：让"描述子匹配 + 原始深度点"这条 fallback 在 covis_mgr == nullptr 时也能跑
// 原实现把阶段 2（描述子 fallback）整段包在 if (covis_mgr != nullptr) 里，
// 而 Tracking::detectAndCorrectLoop() 调 detect(..., nullptr) —— 于是 covis_matches
// 恒为空、Sim3 分支永不进入，回环只能走 ICP/EM。
// 方案 A：阶段 1（confirmed map point）需要 CovisManager，保留在守卫内；
//         阶段 2 只用 keyframe_db_ 里已有的 pts3d + 描述子 matches，提到守卫外。
const fs = require('fs');

const path = 'src/sfc_core/src/loopdetection/loop_detector.cpp';
const lines = fs.readFileSync(path, 'utf8').split('\n');

function findLine(pred, from, label) {
    for (let i = from; i < lines.length; ++i) if (pred(lines[i])) return i;
    throw new Error(`未找到 ${label}`);
}

// ---- 1) 去重集合提到守卫外 ----
const declIdx = findLine(l => l.includes('std::vector<::sfc_core::frontend::CovisMatch> covis_matches;'), 300, 'covis_matches 声明');
const guardIdx = declIdx + 1;
if (lines[guardIdx].trim() !== 'if (covis_mgr != nullptr) {') {
    throw new Error(`守卫行不符：${JSON.stringify(lines[guardIdx])}`);
}
const usedA = findLine(l => l.includes('std::unordered_set<int> used_curr_kpts;'), guardIdx, 'used_curr_kpts');
const usedB = usedA + 1;
if (!lines[usedB].includes('std::unordered_set<int> used_match_kpts;')) {
    throw new Error('used_match_kpts 不在 used_curr_kpts 下一行');
}
lines[usedA] = '        // L2：这两个去重集合提到守卫外（阶段 1 与阶段 2 共用，见下方声明）';
lines.splice(usedB, 1);
lines[usedA] = '';

const guardNow = guardIdx;
lines.splice(guardNow, 0,
    '    // L2：阶段 2（描述子 fallback）在 covis_mgr 为空时也要执行，所以去重集合必须在这里，',
    '    // 两个阶段共用同一份，否则同一个关键点会被重复计入对应点。',
    '    std::unordered_set<int> used_curr_kpts;',
    '    std::unordered_set<int> used_match_kpts;');

const shift = 4;

// ---- 2) 把"阶段 2 + Sim3 块"整段搬到 covis 守卫之外 ----
const stage2Start = findLine(l => l.includes('// ---- 阶段 2：用描述子匹配 + 原始深度点作为 fallback ----'), guardNow, '阶段 2 注释');
const sim3Open = findLine(l => l.includes('if (static_cast<int>(covis_matches.size()) >= 1) {'), stage2Start, 'Sim3 块入口');
const sim3Close = findLine(l => l.trim() === '}' && l.startsWith('        }'), sim3Open + 1, 'Sim3 块闭合');
const guardClose = sim3Close + 1;
if (lines[guardClose] !== '    }') {
    throw new Error(`守卫闭合行不符：${JSON.stringify(lines[guardClose])}`);
}

const block = lines.slice(stage2Start, sim3Close + 1);
// 左移 4 格
const dedented = block.map(l => (l.startsWith('    ') ? l.slice(4) : l));

// 删除原块
lines.splice(stage2Start, sim3Close - stage2Start + 1);
// 现在守卫闭合行上移到 stage2Start
const guardCloseNew = stage2Start;
if (lines[guardCloseNew] !== '    }') {
    throw new Error(`搬移后守卫闭合行不符：${JSON.stringify(lines[guardCloseNew])}`);
}
lines[guardCloseNew] = '    }  // if (covis_mgr != nullptr)：只守阶段 1（L2 后阶段 2 与 Sim3 在守卫外）';

lines.splice(guardCloseNew + 1, 0,
    '',
    '    // L2：下面是阶段 2 —— 用**描述子 matches + keyframe_db_ 里已有的世界坐标点**',
    '    // 构造对应点，不依赖 CovisManager。原实现把它写在 if (covis_mgr != nullptr) 里，',
    '    // 而 Tracking::detectAndCorrectLoop() 传的是 nullptr，于是这段永远不执行、',
    '    // covis_matches 恒空、Sim3 永不进入 —— 回环就只能退化成错的 ICP / 无尺度的 EM。',
    ...dedented);

fs.writeFileSync(path, lines.join('\n'));
console.log(`[L2] ${path}: 阶段 2 + Sim3 块已搬到 covis 守卫外（行 ${guardCloseNew + 4} 起，共 ${dedented.length} 行）`);
