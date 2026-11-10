# 指定CMake最低版本
cmake_minimum_required(VERSION 3.16)

# 【关键】在 project() 之前设置策略，避免自动添加项目名宏
if(POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()

# 项目名
project(sfc_core CXX)

# ==========================================
# 【核心修复】彻底清除 sfc_core 宏定义
# ==========================================
# project(sfc_core CXX) 会自动生成 -Dsfc_core=1 和 -Dsfc_core_EXPORTS，
# 编译器预处理时 sfc_core 被替换为 1，破坏 namespace sfc_core { 语法。
# （INTERFACE 库方案与 ament_export 冲突，改用头文件 #undef 兜底）

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_EXTENSIONS OFF)

# 跨平台 SuiteSparse 配置（兼容 Linux 和 Windows）
if(NOT WIN32)
  list(APPEND CMAKE_MODULE_PATH "/usr/share/cmake-3.22/Modules")
  list(APPEND CMAKE_MODULE_PATH "/usr/lib/cmake/suitesparse")
endif()
find_package(SuiteSparse QUIET)
if(SuiteSparse_FOUND AND NOT TARGET SuiteSparse::SuiteSparse)
  add_library(SuiteSparse::SuiteSparse INTERFACE IMPORTED)
  set_target_properties(SuiteSparse::SuiteSparse PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SuiteSparse_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${SuiteSparse_LIBRARIES}"
  )
endif()

# 跨平台第三方库查找（g2o / Sophus）
#
# 查找路径与选中规则抽到 cmake/ResolveThirdparty.cmake，原因是同一份列表此前在
# sfc_core 与 sfc_ros_humble 各写一遍，一旦不同步就会出现「核心包编过、
# 下游包编不过」且从报错看不出原因（R-023：配置只有一个来源）。
#
# 注意：Linux 上必须走源码编译的那份 g2o。apt 装的 /opt/ros/humble g2o 在本机
# 不完整（g2oTargets.cmake 引用了不存在的 libg2o_ext_freeglut_minimal.so），
# 一旦有 find_package(g2o) 落到它上面就会直接 CMake Error。
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ResolveThirdparty.cmake")
sfc_resolve_g2o_dir()
sfc_resolve_sophus_dir()
# g2oConfig.cmake 内部的 find_dependency 还要能定位到库本体：
# <prefix>/lib/cmake/g2o -> <prefix> 需上溯三层。ResolveThirdparty.cmake 里不写
# 这行，是因为在顶层 CMakeLists 上对普通变量用 PARENT_SCOPE 会报 "no parent scope"。
list(APPEND CMAKE_PREFIX_PATH "${g2o_DIR}/../../..")
find_package(g2o REQUIRED
  COMPONENTS core types_sba types_slam3d types_sim3 solver_dense solver_eigen stuff
)

# D6-b：g2o 的 include 目录只保留一个来源。
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
  get_filename_component(_g2o_prefix "${g2o_DIR}/../../.." ABSOLUTE)
  if(EXISTS "${_g2o_prefix}/include/g2o")
    set(g2o_INCLUDE_DIRS "${_g2o_prefix}/include")
    message(STATUS "[g2o] g2oConfig.cmake 未导出 include，已从 g2o_DIR 推导：${g2o_INCLUDE_DIRS}")
  else()
    message(WARNING
      "[g2o] 既没有导出的 include 目录，也无法从 ${g2o_DIR} 推导出 include/g2o。"
      "编译可能失败，请检查 g2o 安装是否完整。")
  endif()
endif()
if(g2o_INCLUDE_DIRS)
  message(STATUS "[g2o] include: ${g2o_INCLUDE_DIRS}")
endif()

# 跨平台 OpenCV 和 Eigen3
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Sophus（跨平台，尝试多个安装路径）
# Sophus 源码构建后 cmake 配置在 build 目录
if(NOT WIN32)
  list(APPEND CMAKE_PREFIX_PATH "/home/syf/Sophus/build")
endif()
find_package(Sophus QUIET)
if(NOT Sophus_FOUND AND NOT WIN32)
  find_package(Sophus PATHS /usr/local /opt/sophus /usr)
endif()

# Boost for DBoW2 serialization
find_package(Boost REQUIRED COMPONENTS serialization)

# ROS2 基础依赖（ament_export_targets 等函数需要 ament_cmake 提供）
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)

# 包含头文件目录（跨平台）
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/sfc_core
  ${CMAKE_CURRENT_SOURCE_DIR}/include/sfc_core/thirdparty/dbow2
  ${EIGEN3_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${G2O_INCLUDE_DIRS}
)
# Linux: Sophus 可能安装在多个位置
if(NOT WIN32)
  include_directories("/home/syf/g2o_user_install/include")
  # Sophus 源码编译路径（sophus 子目录下无 Sophus/ 子目录）
  # 代码使用 #include "sophus/geometry.hpp"，需加父目录
  include_directories(/home/syf/Sophus)
  # Sophus apt 安装位置
  include_directories(/usr/local/include/sophus)
  include_directories(/usr/local/include)
  include_directories(/usr/include)
endif()
if(Sophus_FOUND AND Sophus_INCLUDE_DIR)
  include_directories(${Sophus_INCLUDE_DIR})
endif()

# 声明库
add_library(${PROJECT_NAME} SHARED
  src/config/config_loader.cpp
  src/frontend/robust_pnp_solver.cpp
  src/frontend/mlpnp_solver.cpp
  src/frontend/covis_core.cpp
  src/frontend/covis_manager.cpp
  src/frontend/tracking.cpp
  src/frontend/keyframe_policy.cpp
  src/frontend/frame.cpp
  src/frontend/sophus_pose_adapter.cpp
  src/feature/bow_adapter.cpp
  src/feature/feature_matcher.cpp
  src/feature/octree_orb.cpp
  # DBoW2 核心（ORB 词袋，用于 SearchByBoW 几何匹配）
  src/thirdparty/dbow2/FORB.cpp
  src/thirdparty/dbow2/BowVector.cpp
  src/thirdparty/dbow2/FeatureVector.cpp
  src/thirdparty/dbow2/ScoringObject.cpp
  # ORB-SLAM3 IMU / EKF
  src/frontend/pose_ekf.cpp
  # 后端存根
  src/mapping/map_point.cpp
  src/mapping/key_frame.cpp
  src/mapping/map.cpp
  src/mapping/local_mapping.cpp
  src/mapping/atlas.cpp
  # 回环检测（.cpp + 全内联头文件 geometric_verifier.hpp / place_recognition.hpp）
  src/loopdetection/loop_detector.cpp
  src/loopdetection/pose_graph_optimizer.cpp
  src/loopdetection/sim3_solver.cpp
  src/loopdetection/sim3_optimizer.cpp
)

# sfc_core 宏在编译参数层清除（头文件内的 #undef sfc_core 是防御性兜底）
# PUBLIC 使其传播到所有消费目标（test_loop_detector、test_visual_slam 等可执行文件）
target_compile_options(${PROJECT_NAME} PUBLIC -Usfc_core)

# 测试可执行程序：回环检测与位姿图优化单元测试
add_executable(test_loop_detector
  test/test_loop_detector.cpp
)
target_link_libraries(test_loop_detector
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  g2o::core g2o::types_sba g2o::types_slam3d g2o::solver_dense g2o::stuff
)
target_compile_options(test_loop_detector PRIVATE -Usfc_core)

# 测试可执行程序：关键帧策略纯逻辑单测（纯逻辑，不需要相机数据与 g2o）
add_executable(test_keyframe_policy
  test/test_keyframe_policy.cpp
)
target_link_libraries(test_keyframe_policy
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
)
target_compile_options(test_keyframe_policy PRIVATE -Usfc_core)
install(TARGETS test_keyframe_policy
  DESTINATION lib/${PROJECT_NAME}
)

# 测试可执行程序：TUM-RGBD 分模块诊断
add_executable(test_tum_module_diag
  test/test_tum_module_diag.cpp
)
target_link_libraries(test_tum_module_diag
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  g2o::core g2o::types_sba g2o::types_slam3d g2o::solver_dense g2o::stuff
)
target_compile_options(test_tum_module_diag PRIVATE -Usfc_core)
install(TARGETS test_tum_module_diag
  DESTINATION lib/${PROJECT_NAME}
)

# 测试可执行程序：TUM-RGBD Benchmark 评估
add_executable(test_tum_rgbd
  test/test_tum_rgbd.cpp
)
target_link_libraries(test_tum_rgbd
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  g2o::core g2o::types_sba g2o::types_slam3d g2o::types_sim3 g2o::solver_dense g2o::stuff
)
target_compile_options(test_tum_rgbd PRIVATE -Usfc_core)
install(TARGETS test_tum_rgbd
  DESTINATION lib/${PROJECT_NAME}
)

# （已移除 test_visual_slam: 依赖不存在的 feature_extractor.hpp）
# 测试可执行程序：ROS2 图像话题 SLAM 跟踪节点
# add_executable(test_visual_slam
#   test/test_visual_slam.cpp
# )
# target_link_libraries(test_visual_slam
#   ${PROJECT_NAME}
#   rclcpp::rclcpp
#   image_transport::image_transport
#   cv_bridge::cv_bridge
#   ${OpenCV_LIBS}
#   g2o::core g2o::types_sba g2o::types_slam3d g2o::solver_dense g2o::stuff
# )
# target_compile_options(test_visual_slam PRIVATE -Usfc_core)

# ==========================================
# 【修复】统一使用关键字签名链接所有依赖
# ==========================================
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    rclcpp::rclcpp
    ${sensor_msgs_TARGETS}
    ${image_transport_TARGETS}
    ${cv_bridge_TARGETS}
  PRIVATE
    Sophus::Sophus
    Eigen3::Eigen
    ${OpenCV_LIBS}
    g2o::core
    g2o::types_sba
    g2o::types_slam3d
    g2o::types_sim3
    g2o::solver_dense
    g2o::solver_eigen
    g2o::stuff
    Boost::serialization
)

# 安装配置
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION include/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

# 安装 config 目录（YAML 配置文件）
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING PATTERN "*.yaml"
)

# mapping 子目录也需要安装（CMake 默认只安装 include/${PROJECT_NAME}/）
install(DIRECTORY include/${PROJECT_NAME}/mapping/
  DESTINATION include/${PROJECT_NAME}/mapping
  FILES_MATCHING PATTERN "*.hpp"
)

# 安装第三方库查找脚本：下游包必须在 find_package(sfc_core) 之前先拿到它，
# 否则 extras 里的 find_package(g2o) / find_package(Sophus) 会落到
# /opt/ros/humble 那份不完整的 g2o 配置上（引用即报错）。
install(FILES cmake/ResolveThirdparty.cmake
  DESTINATION share/${PROJECT_NAME}/cmake
)

# 共享库安装（供其他包链接）
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS test_loop_detector
  DESTINATION lib/${PROJECT_NAME}
)

# （test_visual_slam 已注释，参考上方）


ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 OpenCV Sophus g2o rclcpp sensor_msgs image_transport cv_bridge)

# 测试配置
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()