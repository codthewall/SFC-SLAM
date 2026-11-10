// 轻量花括号/圆括号配平检查（无 C++ 工具链时的兜底静态检查）。
// 用法: node scripts/check_braces.js <源文件>
const fs = require('fs');

const path = process.argv[2];
if (!path) {
  console.error('usage: node scripts/check_braces.js <file>');
  process.exit(2);
}

const lines = fs.readFileSync(path, 'utf8').split(/\r?\n/);
let brace = 0;
let paren = 0;
let brack = 0;
let inBlockComment = false;
let inString = null;
let escaped = false;
const negatives = [];

for (let i = 0; i < lines.length; i++) {
  const line = lines[i];
  let inLineComment = false;
  for (let j = 0; j < line.length; j++) {
    const c = line[j];
    const n = line[j + 1];

    if (inLineComment) break;
    if (inBlockComment) {
      if (c === '*' && n === '/') {
        inBlockComment = false;
        j++;
      }
      continue;
    }
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (c === '\\') {
        escaped = true;
      } else if (c === inString) {
        inString = null;
      }
      continue;
    }
    if (c === '/' && n === '/') {
      inLineComment = true;
      break;
    }
    if (c === '/' && n === '*') {
      inBlockComment = true;
      j++;
      continue;
    }
    if (c === '"' || c === "'") {
      inString = c;
      continue;
    }
    if (c === '{') brace++;
    else if (c === '}') {
      brace--;
      if (brace < 0) negatives.push(i + 1);
    } else if (c === '(') paren++;
    else if (c === ')') paren--;
    else if (c === '[') brack++;
    else if (c === ']') brack--;
  }
}

console.log(`file=${path}`);
console.log(`final brace=${brace} paren=${paren} brack=${brack}`);
console.log(`first_negative_brace_line=${negatives.length ? negatives[0] : 'none'}`);
process.exit(brace === 0 && paren === 0 && brack === 0 && negatives.length === 0 ? 0 : 1);
