// E6 静态自检：括号平衡 + 关键符号一致性 + 遗留引用扫描
// 环境无 C++ 工具链，只能做文本级检查。
const fs = require('fs');
const path = require('path');

const files = [
  'src/sfc_core/src/frontend/tracking.cpp',
  'src/sfc_core/include/sfc_core/frontend/tracking.hpp',
  'src/sfc_core/src/frontend/robust_pnp_solver.cpp',
  'src/sfc_core/include/sfc_core/frontend/robust_pnp_solver.hpp',
  'src/sfc_core/src/frontend/covis_manager.cpp',
];

function stripNoise(src) {
  // 去掉注释与字符串字面量，避免括号/关键字误判
  let out = '';
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    const d = src[i + 1];
    if (c === '/' && d === '/') {
      while (i < n && src[i] !== '\n') i++;
      continue;
    }
    if (c === '/' && d === '*') {
      i += 2;
      while (i < n && !(src[i] === '*' && src[i + 1] === '/')) i++;
      i += 2;
      continue;
    }
    if (c === '"') {
      i++;
      while (i < n && src[i] !== '"') { if (src[i] === '\\') i++; i++; }
      i++;
      out += '""';
      continue;
    }
    if (c === "'") {
      i++;
      while (i < n && src[i] !== "'") { if (src[i] === '\\') i++; i++; }
      i++;
      out += "''";
      continue;
    }
    out += c;
    i++;
  }
  return out;
}

let failed = false;
for (const f of files) {
  const src = fs.readFileSync(f, 'utf8');
  const code = stripNoise(src);
  const pairs = { '(': ')', '[': ']', '{': '}' };
  const stack = [];
  let line = 1;
  let bad = null;
  for (let i = 0; i < code.length; ++i) {
    const ch = code[i];
    if (ch === '\n') line++;
    if (pairs[ch]) stack.push({ ch, line });
    else if (ch === ')' || ch === ']' || ch === '}') {
      const top = stack.pop();
      if (!top) { bad = `多余的 ${ch} 在行 ${line}`; break; }
      if (pairs[top.ch] !== ch) { bad = `行 ${line} 的 ${ch} 与行 ${top.line} 的 ${top.ch} 不匹配`; break; }
    }
  }
  if (!bad && stack.length) bad = `未闭合 ${stack.map(s => s.ch + '@' + s.line).join(',')}`;
  console.log(`${bad ? 'FAIL' : 'OK  '} ${f} (${src.split('\n').length} 行)${bad ? ' -> ' + bad : ''}`);
  if (bad) failed = true;
}

// 遗留引用扫描
const trackingSrc = fs.readFileSync('src/sfc_core/src/frontend/tracking.cpp', 'utf8');
const leftovers = [
  ['observed_count', /->observed_count|\bmp\.observed_count/],
  ['R_eigen', /\bR_eigen\b/],
  ['mlpTemporalPoints_.clear()', /mlpTemporalPoints_\.clear\(\)/],
  ['need_undistort', /need_undistort/],
];
for (const [name, re] of leftovers) {
  const hits = trackingSrc.split('\n')
    .map((l, i) => [i + 1, l])
    .filter(([, l]) => re.test(l));
  console.log(`${hits.length ? 'WARN' : 'OK  '} 遗留 ${name}: ${hits.length} 处` +
    (hits.length ? ' -> ' + hits.map(([ln]) => ln).join(',') : ''));
}

process.exit(failed ? 1 : 0);
