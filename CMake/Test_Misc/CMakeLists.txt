set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Test_Misc)

set(SRC_MATH
    ${CURRENT_SOURCE_DIR}/T_RayCPU.cpp
    ${CURRENT_SOURCE_DIR}/T_RayGPU.cu
    ${CURRENT_SOURCE_DIR}/T_MatrixCPU.cpp
    ${CURRENT_SOURCE_DIR}/T_MatrixGPU.cu
    ${CURRENT_SOURCE_DIR}/T_Quaternion.cpp
    ${CURRENT_SOURCE_DIR}/T_VectorCPU.cpp
    ${CURRENT_SOURCE_DIR}/T_VectorGPU.cu)

set(SRC_SCENE_IO
    ${CURRENT_SOURCE_DIR}/T_GPUScene.cpp
    ${CURRENT_SOURCE_DIR}/T_SceneIO.cpp
    ${CURRENT_SOURCE_DIR}/T_SceneIOCommon.cpp)

set(SRC_TRACER_MOCK
    ${CURRENT_SOURCE_DIR}/SimpleTracerSetup.h
    ${CURRENT_SOURCE_DIR}/T_OptiX.cpp
    ${CURRENT_SOURCE_DIR}/T_AOTracers.cpp
    ${CURRENT_SOURCE_DIR}/T_MockTracer.cu
    ${CURRENT_SOURCE_DIR}/T_PathTracers.cpp
    ${CURRENT_SOURCE_DIR}/T_PPGTracers.cpp
    ${CURRENT_SOURCE_DIR}/T_SelfNode.cpp
    ${CURRENT_SOURCE_DIR}/T_SimpleTracers.cpp)

set(SRC_COMMON ${CURRENT_SOURCE_DIR}/main.cpp)

set(SRC_ALL
    ${SRC_MATH}
    ${SRC_SCENE_IO}
    ${SRC_TRACER_MOCK}
    ${SRC_COMMON})

source_group("Math" FILES ${SRC_MATH})
source_group("SceneIO" FILES ${SRC_SCENE_IO})
source_group("TracerMock" FILES ${SRC_TRACER_MOCK})
source_group("" FILES ${SRC_COMMON})

add_executable(Test_Misc ${SRC_ALL})

target_link_libraries(Test_Misc PRIVATE
                      RayLib
                      VisorGL
                      gtest::gtest
                      gtest::gmock
                      mray::meta_compile_opts
                      mray::cuda_extra_compile_opts)

set_target_properties(Test_Misc PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Not-linked but dependent targets
add_dependencies(Test_Misc TracerCUDA)

if(MSVC)
    set(DEBUG_ARGS "--gtest_filter=*.* --gtest_catch_exceptions=0 --gtest_repeat=1 --gtest_shuffle=0")

    set_target_properties(Test_Misc PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})

    # Copy DLLS
    add_custom_command(TARGET Test_Misc POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${MRAY_CONFIG_LIB_DIRECTORY}/gmock.dll"
                       "${MRAY_CONFIG_LIB_DIRECTORY}/gtest.dll"
                       $<$<CONFIG:Debug>:"${MRAY_CONFIG_LIB_DIRECTORY}/gmock.pdb">
                       $<$<CONFIG:Debug>:"${MRAY_CONFIG_LIB_DIRECTORY}/gtest.pdb">
                       ${MRAY_CONFIG_BIN_DIRECTORY})
endif()