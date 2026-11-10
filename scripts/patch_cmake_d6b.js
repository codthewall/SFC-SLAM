// D6-b：g2o 的 include 目录不再硬编码两份安装路径。
//
// 原代码（src/sfc_core/CMakeLists.txt 的 `if(NOT WIN32)` 段）：
//   set(_g2o_user_include "/home/syf/g2o_user_install/include")
//   set(_g2o_types_include "/home/syf/g2o/build/install/include")
//   set(G2O_INCLUDE_DIRS "${g2o_INCLUDE_DIRS};${_g2o_user_include};${_g2o_types_include}" CACHE PATH ...)
//
// 两个问题：
//   1. 把两台机器上不同时期的 g2o 安装同时塞进搜索路径。两份 g2o 头文件版本不同时，
//      编译器按顺序取第一个命中的 —— 谁赢取决于路径顺序，而不是配置，属于典型的
//      "同一份代码在两台机器编出不同行为"（R-023：配置只有一个来源）。
//   2. ResolveThirdparty.cmake 已经把 g2o_DIR 唯一化了；这里再手工拼 include 路径，
//      等于绕过那一层，等于又有第二个来源。
//
// 改成：直接使用 find_package(g2o) 导出的 include 目录；若它为空，才从 g2o_DIR
// 反推（<prefix>/lib/cmake/g2o → <prefix>/include），并在 STATUS 里说明来源。
const fs = require('fs');

const p = 'src/sfc_core/CMakeLists.txt';
let s = fs.readFileSync(p, 'utf8');

const oldBlock = `if(NOT WIN32)
  # 手动强制设置 g2o include 目录（g2o 的 config 未完整导出 include 路径）
  set(_g2o_user_include "/home/syf/g2o_user_install/include")
  set(_g2o_types_include "/home/syf/g2o/build/install/include")
  set(G2O_INCLUDE_DIRS "\${g2o_INCLUDE_DIRS};\${_g2o_user_include};\${_g2o_types_include}" CACHE PATH "g2o include directory")
endif()`;

const newBlock = `# D6-b：g2o 的 include 目录只保留一个来源。
#
# 原写法在 Linux 上把两份 g2o 安装同时拼进 G2O_INCLUDE_DIRS：
#   /home/syf/g2o_user_install/include 与 /home/syf/g2o/build/install/include
# 两份头文件版本不同时，编译器按路径顺序取第一个命中的 —— 谁赢取决于这里的书写
# 顺序，而不是任何配置，属于"同一份代码在两台机器编出不同行为"（R-023）。
# 而且 ResolveThirdparty.cmake 已经把 g2o_DIR 唯一化了，这里再手工拼一遍
# 等于绕过那一层、又造出第二个来源。
#
# 现在：优先用 find_package(g2o) 导出的 include；导出为空时从 g2o_DIR 反推
# （<prefix>/lib/cmake/g2o → <prefix>/include），并把结果写进 STATUS 以便核对。
if(NOT g2o_INCLUDE_DIRS)
  get_filename_component(_g2o_prefix "\${g2o_DIR}/../../.." ABSOLUTE)
  if(EXISTS "\${_g2o_prefix}/include/g2o")
    set(g2o_INCLUDE_DIRS "\${_g2o_prefix}/include")
    message(STATUS "[g2o] g2oConfig.cmake 未导出 include，已从 g2o_DIR 推导：\${g2o_INCLUDE_DIRS}")
  else()
    message(WARNING
      "[g2o] 既没有导出的 include 目录，也无法从 \${g2o_DIR} 推导出 include/g2o。"
      "编译可能失败，请检查 g2o 安装是否完整。")
  endif()
endif()
if(g2o_INCLUDE_DIRS)
  message(STATUS "[g2o] include: \${g2o_INCLUDE_DIRS}")
endif()`;

if (s.indexOf(oldBlock) < 0) throw new Error('未找到 g2o include 硬编码块');
s = s.replace(oldBlock, newBlock);
fs.writeFileSync(p, s);
console.log('[D6-b] g2o include 目录改为单一来源');
