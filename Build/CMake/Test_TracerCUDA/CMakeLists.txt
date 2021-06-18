set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Test_TracerCUDA)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/main.cpp
    ${CURRENT_SOURCE_DIR}/T_DTree.cu
    ${CURRENT_SOURCE_DIR}/T_ParallelReduction.cu
    ${CURRENT_SOURCE_DIR}/T_ParallelScan.cu
    ${CURRENT_SOURCE_DIR}/T_PiecewiseDistribution.cu
    ${CURRENT_SOURCE_DIR}/T_RandomGPU.cu
    ${CURRENT_SOURCE_DIR}/T_STree.cu
    ${CURRENT_SOURCE_DIR}/T_Texture.cu)

source_group("" FILES ${SRC_COMMON})

# Platform Specific Libraries
if(MSVC)
    set(PLATFORM_SPEC_LIBRARIES)
elseif(UNIX)
    set(PLATFORM_SPEC_LIBRARIES tbb)
endif()

# Exe
add_executable(Test_TracerCUDA ${SRC_COMMON})

target_include_directories(Test_TracerCUDA PRIVATE
                           ${MRAY_LIB_INCLUDE_DIRECTORY}
                           ${MRAY_SOURCE_DIRECTORY})

target_compile_definitions(Test_TracerCUDA PRIVATE ${MRAY_PREPROCESSOR_DEFS_CUDA_GENERIC})

target_compile_options(Test_TracerCUDA PRIVATE ${MRAY_COMPILER_FLAGS_GENERIC})

set_target_properties(Test_TracerCUDA PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(Test_TracerCUDA PRIVATE
                      RayLib
                      TracerCUDA_ForTest
                      $<$<CONFIG:Debug>:gtestd>
                      $<$<CONFIG:Debug>:gmockd>
                      $<$<CONFIG:Release>:gtest>
                      $<$<CONFIG:Release>:gmock>
                      ${PLATFORM_SPEC_LIBRARIES})

if(MSVC)

    set(DEBUG_ARGS "--gtest_filter=*.* --gtest_catch_exceptions=0 --gtest_repeat=1 --gtest_shuffle=0")

    set_target_properties(Test_TracerCUDA PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()