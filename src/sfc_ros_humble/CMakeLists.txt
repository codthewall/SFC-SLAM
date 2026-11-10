cmake_minimum_required(VERSION 3.8)
# 【关键】包名必须和目录名完全一致：sfc_ros_humble
project(sfc_ros_humble)

# ========== 基础编译配置 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pthread")

# ========== 查找依赖库 ==========
# ROS2核心依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# ========== 第三方库查找路径（必须在 find_package(sfc_core) 之前） ==========
# sfc_core 的 ament 导出文件（ament_cmake_export_dependencies-extras.cmake）里
# 有几句裸的 find_package(g2o) / find_package(Sophus)。只要它们先跑到，就会落到
# /opt/ros/humble 那份 g2o 上，直接 CMake Error：
#   The imported target "g2o::freeglut_minimal" references the file
#   "/opt/ros/humble/lib/x86_64-linux-gnu/libg2o_ext_freeglut_minimal.so"
#   but this file does not exist.
# 原因：g2o_DIR 是 per-build-directory 的 CMakeCache 变量，sfc_core 里设过
# 并不会传到本包。所以这里先把 g2o_DIR 指向源码编译的那份（写进 CACHE），
# 后面 extras 里的 find_package(g2o) 会复用同一个 g2o_DIR，不再重新搜索。
#
# 查找规则与 sfc_core 共用同一份 ResolveThirdparty.cmake，避免两处列表不同步
# （R-023：配置只有一个来源）。
#
# 三个候选按可用性排序：
#   1) 同在工作空间 src/ 下 —— 一定有，也是 colcon 构建时的常态；
#   2) sfc_core 已安装出来的那份（build 目录相对工作空间根为 ../../install）；
#   3) AMENT_PREFIX_PATH 里任意 prefix 下的 sfc_core 安装 —— 本包被别的工作空间复用时用。
#
# 注意：不能写 "${sfc_core_DIR}/...". 这一刻 find_package(sfc_core) 还没跑，
# sfc_core_DIR 是空的，展开出来就是 "/..."。
set(_sfc_resolve_3rd "")
foreach(_cand
    "${CMAKE_CURRENT_SOURCE_DIR}/../sfc_core/cmake/ResolveThirdparty.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../install/sfc_core/share/sfc_core/cmake/ResolveThirdparty.cmake")
  if(EXISTS "${_cand}")
    set(_sfc_resolve_3rd "${_cand}")
    break()
  endif()
endforeach()

if(NOT _sfc_resolve_3rd AND NOT WIN32)
  # AMENT_PREFIX_PATH 在 Linux 是 ':' 分隔的 prefix 列表（colcon 会带上 install/ 与 /opt/ros/humble）。
  # Windows 上该变量本身是 ';' 分隔，按 ':' 切会切坏盘符（C:/...），故只在非 Windows 做这一步。
  string(REPLACE ":" ";" _sfc_prefixes "$ENV{AMENT_PREFIX_PATH}")
  foreach(_p IN LISTS _sfc_prefixes)
    if(EXISTS "${_p}/share/sfc_core/cmake/ResolveThirdparty.cmake")
      set(_sfc_resolve_3rd "${_p}/share/sfc_core/cmake/ResolveThirdparty.cmake")
      break()
    endif()
  endforeach()
endif()

if(_sfc_resolve_3rd)
  include("${_sfc_resolve_3rd}")
  sfc_resolve_g2o_dir()
  sfc_resolve_sophus_dir()
  list(APPEND CMAKE_PREFIX_PATH "${g2o_DIR}/../../..")
else()
  # 不直接 FATAL：sfc_core 也可能装在别的 prefix、路径关系不同。
  # 但必须把风险说清楚，否则会退回本次那个「引用不存在的 .so」的报错，
  # 而报错里完全看不出是 g2o 路径问题。
  message(WARNING
    "[sfc_ros_humble] 找不到 ResolveThirdparty.cmake（期望在 "
    "${CMAKE_CURRENT_SOURCE_DIR}/../sfc_core/cmake/ResolveThirdparty.cmake）。\n"
    "接下来 find_package(sfc_core) 会连带执行 find_package(g2o)，"
    "若落到 /opt/ros/humble 那份 g2o 上，会报 "
    "'g2o::freeglut_minimal references ... libg2o_ext_freeglut_minimal.so ... does not exist'。\n"
    "遇到该报错时，用 -Dg2o_DIR=<你的g2o>/lib/cmake/g2o 显式指定。")
endif()

# 依赖我们自己的核心算法包 sfc_core
find_package(sfc_core REQUIRED)
find_package(OpenCV REQUIRED)

# ========== 头文件路径 ==========
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

# ========== 节点源码声明（后续加节点时取消注释） ==========
# set(ROS_NODE_SRC
#   src/camera_input_node.cpp
#   src/slam_node.cpp
# )

# ========== 编译可执行节点（有源码时取消注释） ==========
# add_executable(sfc_slam_node ${ROS_NODE_SRC})

# ========== 链接依赖库（有源码时取消注释） ==========
# ament_target_dependencies(sfc_slam_node
#   rclcpp
#   sensor_msgs
#   nav_msgs
#   geometry_msgs
#   visualization_msgs
#   cv_bridge
#   image_transport
#   tf2_ros
#   tf2_geometry_msgs
#   sfc_core
# )
# target_link_libraries(sfc_slam_node
#   ${OpenCV_LIBS}
# )

# ========== 安装规则 ==========
# 安装可执行节点（有源码时取消注释）
# install(
#   TARGETS sfc_slam_node
#   RUNTIME DESTINATION lib/${PROJECT_NAME}
# )

# 安装头文件：include/sfc_ros_humble/
# git 不跟踪空目录：本包还没有对外头文件时，clone 下来这个目录根本不存在，
# install(DIRECTORY ...) 会报 "file INSTALL cannot find ...: No such file or directory"。
# launch/ 与 rviz/ 同理，所以三处都加 EXISTS 守卫；目录里有东西时行为不变。
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}")
  install(
    DIRECTORY include/${PROJECT_NAME}/
    DESTINATION include/${PROJECT_NAME}
  )
endif()

# 安装launch文件
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/launch")
  install(
    DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
  )
endif()

# 安装RViz配置文件
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rviz")
  install(
    DIRECTORY rviz/
    DESTINATION share/${PROJECT_NAME}/rviz
  )
endif()

# ========== ROS2必须的结尾 ==========
ament_package()