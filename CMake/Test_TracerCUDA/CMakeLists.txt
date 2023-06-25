set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Test_TracerCUDA)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/main.cpp
    ${CURRENT_SOURCE_DIR}/T_DTree.cu
    ${CURRENT_SOURCE_DIR}/T_ParallelReduction.cu
    ${CURRENT_SOURCE_DIR}/T_ParallelScan.cu
    ${CURRENT_SOURCE_DIR}/T_PiecewiseDistribution.cu
    ${CURRENT_SOURCE_DIR}/T_RandomGPU.cu
    ${CURRENT_SOURCE_DIR}/T_STree.cu
    ${CURRENT_SOURCE_DIR}/T_Texture.cu
    ${CURRENT_SOURCE_DIR}/T_AnisoSVO.cu
    ${CURRENT_SOURCE_DIR}/T_ProductSampler.cu
    ${CURRENT_SOURCE_DIR}/T_BlockSegmentedReduce.cu
    ${CURRENT_SOURCE_DIR}/T_BlockSegmentedScan.cu
    ${CURRENT_SOURCE_DIR}/T_BlockPWCDistribution.cu
    ${CURRENT_SOURCE_DIR}/T_BlockPWLDistribution.cu
    ${CURRENT_SOURCE_DIR}/T_ReconFilter.cu)

source_group("" FILES ${SRC_COMMON})

# Exe
add_executable(Test_TracerCUDA ${SRC_COMMON})

target_link_libraries(Test_TracerCUDA PRIVATE
                      RayLib
                      VisorGL
                      ImageIO
                      TracerCUDA_Static
                      gtest::gtest
                      gtest::gmock
                      mray::meta_compile_opts
                      mray::cuda_extra_compile_opts)


set_target_properties(Test_TracerCUDA PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(MSVC)

    set(DEBUG_ARGS "--gtest_filter=*.* --gtest_catch_exceptions=0 --gtest_repeat=1 --gtest_shuffle=0")

    set_target_properties(Test_TracerCUDA PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})

    # Copy DLLS
    add_custom_command(TARGET Test_TracerCUDA POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${MRAY_CONFIG_LIB_DIRECTORY}/gmock.dll"
                       "${MRAY_CONFIG_LIB_DIRECTORY}/gtest.dll"
                       $<$<CONFIG:Debug>:"${MRAY_CONFIG_LIB_DIRECTORY}/gmock.pdb">
                       $<$<CONFIG:Debug>:"${MRAY_CONFIG_LIB_DIRECTORY}/gtest.pdb">
                       ${MRAY_CONFIG_BIN_DIRECTORY})

endif()