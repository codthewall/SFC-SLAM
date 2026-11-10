# 在 project(g2o) 调用前执行，清除任何外来的 sfc_core 宏
# 并确保 PROJECT_NAME 为 g2o_core，使 g2o cmake 文件使用正确的项目名前缀
if(PROJECT_NAME STREQUAL "sfc_core" OR DEFINED CMAKE_PROJECT_sfc_core_INCLUDE)
  # 清除 sfc_core 预处理器宏（来自父项目）
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E echo "Removing -Dsfc_core=1 from CMAKE_CXX_FLAGS"
      RESULT_VARIABLE _result
    )
  endif()
endif()

# 强制将 PROJECT_NAME 设置为 g2o_core
# 这样 g2o 的 configure_file() 会生成 g2o_coreConfig.cmake 等文件
set(PROJECT_NAME "g2o_core" CACHE INTERNAL "" FORCE)
