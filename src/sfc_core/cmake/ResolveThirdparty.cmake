# ================================================================
# ResolveThirdparty.cmake —— 非 ROS 第三方库查找路径的单一来源
# ================================================================
#
# 为什么需要这个文件
# ------------------
# CMake 的 CMAKE_PREFIX_PATH 与 g2o_DIR / Sophus_DIR 都是 **per-build-directory**
# 的变量：在 sfc_core 的 CMakeLists 里设过，不会传递到 sfc_ros_humble 的
# build 目录。
#
# 但 sfc_core 通过 ament 导出了依赖：
#
#   ament_export_dependencies(Eigen3 OpenCV Sophus g2o rclcpp ...)
#
# 于是 sfc_ros_humble 执行 find_package(sfc_core) 时，会连带执行
#   install/sfc_core/share/sfc_core/cmake/ament_cmake_export_dependencies-extras.cmake
# 里那几句**裸的** find_package(Sophus REQUIRED) / find_package(g2o REQUIRED)。
# 下游的 build 目录里没有这些路径，就会去找系统/ROS 前缀下的那份：
#
#   * g2o：/opt/ros/humble 那份是不完整的，导出的 g2oTargets.cmake 引用了
#     并不存在的 libg2o_ext_freeglut_minimal.so，一 include 就 CMake Error：
#       The imported target "g2o::freeglut_minimal" references the file
#       ".../libg2o_ext_freeglut_minimal.so" but this file does not exist.
#   * Sophus：apt/ROS 前缀下通常没有，或版本与 /home/syf/Sophus/build 不一致。
#
# 解决办法：把这个「候选路径 + 选中规则」抽到本文件，两个包都在
# **find_package(sfc_core) 之前**调用一次。设进去的是 CMakeCache 变量
# （<Pkg>_DIR），extras 里后续那句 find_package 会直接复用，不再重新搜索。
#
# 为什么不让两个包各写一份候选列表
# --------------------------------
# 两处列表一旦不同步，就会出现「核心包能编、下游包编不过」或者反过来，而报错
# 里完全看不出是路径问题（R-023：配置只有一个来源）。
#
# 用法
# ----
#   include("<repo>/src/sfc_core/cmake/ResolveThirdparty.cmake")
#   sfc_resolve_g2o_dir()        # 必须：设置 g2o_DIR（CACHE）
#   sfc_resolve_sophus_dir()     # 必须：设置 Sophus_DIR 或追加 CMAKE_PREFIX_PATH
#   # 再补一句，让各 config 内部的 find_dependency 能定位到库本体：
#   list(APPEND CMAKE_PREFIX_PATH "${g2o_DIR}/../../..")
#   find_package(g2o REQUIRED COMPONENTS ...)   # 组件由调用方决定
#
# 说明：两个函数内部**不**修改调用方的 CMAKE_PREFIX_PATH。函数里对普通变量
# 用 PARENT_SCOPE，在顶层目录会触发 "current scope has no parent"，所以那一步
# 留给调用方显式写，反而更清楚。
# ================================================================

# g2oConfig.cmake 里有 find_dependency(OpenGL)，会让 CMake 报
# CMP0072（GLVND vs legacy GL 二选一）的 dev warning。该警告与我们的构建
# 无关，但每次配置刷一屏。
#
# 两种写法都要有，缺一不可：
#   - 设 OpenGL_GL_PREFERENCE 变量：FindOpenGL.cmake 直接查这个变量，是变量
#     作用域，一定会被 find_dependency 那条链看到。
#   - 设 CMP0072 policy：仅在本目录作用域生效。find_package() 在 CMake 3.12+
#     会为 config mode 开独立 policy scope，所以它**不一定**传进
#     sfc_coreConfig.cmake 里那次 find_dependency —— 这也是只写 policy 时
#     警告仍然出现的原因。保留它是为了在本目录内直接 find_package(g2o) 的
#     情况（sfc_core 自身构建）。
if(NOT DEFINED OpenGL_GL_PREFERENCE)
  set(OpenGL_GL_PREFERENCE GLVND)
endif()
if(POLICY CMP0072)
  cmake_policy(SET CMP0072 NEW)
endif()

# ----------------------------------------------------------------
# 设置 g2o_DIR 指向可用的 g2o 配置目录；找不到则 FATAL_ERROR（说明原因）
# ----------------------------------------------------------------
function(sfc_resolve_g2o_dir)
  # 1. 已经缓存了 g2o_DIR：尊重它，但**必须排除 /opt/ros**。
  #    不能只判 "g2oConfig.cmake 存在"：apt 那份 config 是存在的，只是它引用的
  #    libg2o_ext_freeglut_minimal.so 不存在 —— 正是本次故障本身。
  #    另外，上一次失败的构建会把 /opt/ros/... 写进 CMakeCache（find_package 在
  #    定位到 config 后、include 失败前就会写 <pkg>_DIR），若不排除，重新配置
  #    时会走这个提前 return，等于修复从未生效。
  if(g2o_DIR AND EXISTS "${g2o_DIR}/g2oConfig.cmake")
    if(g2o_DIR MATCHES "^/opt/ros/")
      message(STATUS
        "[ResolveThirdparty] 忽略缓存里的 g2o_DIR=${g2o_DIR}（ROS 自带的 g2o 缺 "
        "libg2o_ext_freeglut_minimal.so，include 其 targets 文件即报错），改查源码编译版本")
    else()
      message(STATUS "[ResolveThirdparty] 沿用 g2o_DIR=${g2o_DIR}")
      return()
    endif()
  endif()

  if(WIN32)
    set(_sfc_g2o_candidates
        "C:/g2o/install/lib/cmake/g2o"
        "C:/g2o/build/install/lib/cmake/g2o"
        "C:/Program Files/g2o/lib/cmake/g2o"
        "C:/Program Files (x86)/g2o/lib/cmake/g2o")
  else()
    # Linux：只用源码编译的 g2o。/opt/ros/humble 那份缺 .so，不能作为回退。
    set(_sfc_g2o_candidates
        "/home/syf/g2o/build/install/lib/cmake/g2o"
        "/home/syf/g2o_user_install/lib/cmake/g2o"
        "/usr/local/lib/cmake/g2o"
        "/usr/lib/x86_64-linux-gnu/cmake/g2o")
  endif()

  foreach(_cand IN LISTS _sfc_g2o_candidates)
    if(EXISTS "${_cand}/g2oConfig.cmake")
      set(g2o_DIR "${_cand}" CACHE PATH "g2o cmake config directory" FORCE)
      message(STATUS "[ResolveThirdparty] g2o: ${_cand}")
      return()
    endif()
  endforeach()

  message(FATAL_ERROR
    "[ResolveThirdparty] 未找到可用的 g2o。已尝试：\n"
    "  /home/syf/g2o/build/install/lib/cmake/g2o/g2oConfig.cmake\n"
    "  /home/syf/g2o_user_install/lib/cmake/g2o/g2oConfig.cmake\n"
    "  /usr/local/lib/cmake/g2o/g2oConfig.cmake\n"
    "注意：/opt/ros/humble 下的 g2o 在本机不完整"
    "（缺 libg2o_ext_freeglut_minimal.so），不能作为回退。\n"
    "可运行：cd ~/g2o/build && cmake .. -DCMAKE_INSTALL_PREFIX=./install "
    "&& make -j$(nproc) && make install\n"
    "或手动指定：-Dg2o_DIR=/path/to/g2o/lib/cmake/g2o")
endfunction()

# ----------------------------------------------------------------
# 让 find_package(Sophus) 能找到 - sfc_core 用的是源码树 build 目录
# ----------------------------------------------------------------
# 与 g2o 不同，Sophus 在 sfc_core 里是 `find_package(Sophus QUIET)` + PATHS 兜底，
# 所以这里只要把候选目录追加进 CMAKE_PREFIX_PATH（CACHE 形式）即可，
# 不需要强制 <Pkg>_DIR。用 CACHE 是为了让 extras 里那句 find_package(Sophus REQUIRED)
# 也能看到。
function(sfc_resolve_sophus_dir)
  if(WIN32)
    return()
  endif()

  if(Sophus_DIR AND EXISTS "${Sophus_DIR}/SophusConfig.cmake")
    message(STATUS "[ResolveThirdparty] 沿用 Sophus_DIR=${Sophus_DIR}")
    return()
  endif()

  # 注意 Sophus 的 include 布局：代码写的是 #include "sophus/geometry.hpp"，
  # 需要的是包含 sophus/ 子目录的那个前缀（源码树里是 /home/syf/Sophus 本身）。
  foreach(_cand
      "/home/syf/Sophus/build"
      "/usr/local"
      "/usr"
      "/opt/sophus")
    if(EXISTS "${_cand}/SophusConfig.cmake" OR EXISTS "${_cand}/lib/cmake/Sophus/SophusConfig.cmake")
      set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${_cand}" CACHE STRING "" FORCE)
      message(STATUS "[ResolveThirdparty] Sophus prefix: ${_cand}")
      return()
    endif()
  endforeach()

  # 找不到 config 也要继续：sfc_core 本身对 Sophus 是 QUIET + PATHS 兜底，
  # 且下游 extras 里那句 REQUIRED 若真的需要，会在那里给出更准确的报错。
  message(STATUS "[ResolveThirdparty] 未找到 Sophus 的 cmake config，交由 find_package(Sophus) 自行处理")
endfunction()
