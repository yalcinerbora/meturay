set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Test_Misc)

set(SRC_MATH
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

target_include_directories(Test_Misc PRIVATE
                           ${MRAY_LIB_INCLUDE_DIRECTORY}
                           ${MRAY_SOURCE_DIRECTORY})

target_compile_definitions(Test_Misc PRIVATE ${MRAY_PREPROCESSOR_DEFS_CUDA_GENERIC})

target_compile_options(Test_Misc PRIVATE ${MRAY_COMPILER_FLAGS_GENERIC})

set_target_properties(Test_Misc PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(Test_Misc PRIVATE
                      RayLib
                      VisorGL
                      $<$<CONFIG:Debug>:gtestd>
                      $<$<CONFIG:Debug>:gmockd>
                      $<$<CONFIG:Release>:gtest>
                      $<$<CONFIG:Release>:gmock>)

if(MSVC)

    set(DEBUG_ARGS "--gtest_filter=*.* --gtest_catch_exceptions=0 --gtest_repeat=1 --gtest_shuffle=0")

    set_target_properties(Test_Misc PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})

    # Copy DLLS
    add_custom_command(TARGET Test_Misc POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${MRAY_CONFIG_LIB_DIRECTORY}/gmock$<$<CONFIG:Debug>:d>.dll"
                       "${MRAY_CONFIG_LIB_DIRECTORY}/gtest$<$<CONFIG:Debug>:d>.dll"
                       ${MRAY_CONFIG_BIN_DIRECTORY})
endif()