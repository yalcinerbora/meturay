# Modified Version of this
# https://github.com/NVIDIA/OptiX_Apps/blob/master/3rdparty/CMake/nvcuda_compile_ptx.cmake

# Generate a custom build rule to translate *.cu files to *.ptx files.
# NVCUDA_COMPILE_PTX(
#   MAIN_TARGET TargetName
#   SOURCES file1.cu file2.cu ...
#   GENERATED_TARGET <generated target, (variable stores TargetName_Optix)>
#   EXTRA_OPTIONS <opions for nvcc> ...
# )

# Generates *.ptx files for the given source files.
# Unlike the copied code;
#    It also generates a custom target for files since I did not want to see
#    PTX output on the Visual Studio.
#
#    It returns the generated target for depedency setting.
#
#    It tries to set the dependencies automatically (dunno if this works tho)
#
#    Additionally it generates different PTX for each Compute Capability defined
#    in CMAKE_CUDA_ARCHITECTURES variable
#
#    Finally It also outputs as <filename>_CC[50,61..].o.ptx
#    because it is a good pun =) (Normally it was goint to be optx but maybe some
#    files)

FUNCTION(NVCC_COMPILE_PTX)
    set(oneValueArgs GENERATED_TARGET MAIN_TARGET)
    set(multiValueArgs EXTRA_OPTIONS SOURCES)

    CMAKE_PARSE_ARGUMENTS(NVCC_COMPILE_PTX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Add -- ptx and extra provided options
    # Main Compile Options as well form the system
    set(NVCC_COMPILE_OPTIONS "")
    #get_directory_property(ALL_COMPILE_OPTS COMPILE_OPTIONS)
    #set(ALL_COMPILE_DEFS ${MRAY_PREPROCESSOR_DEFS_CUDA_GENERIC})

    message(STATUS ${ALL_COMPILE_DEFS})

    list(APPEND NVCC_COMPILE_OPTIONS ${NVCC_COMPILE_PTX_EXTRA_OPTIONS}
        --ptx
        --machine=64

        "-std=c++${CMAKE_CUDA_STANDARD}"
        "--relocatable-device-code=true"
        "-I${OPTIX_INCLUDE_DIR}"
        "-I${MRAY_SOURCE_DIRECTORY}"
        "-I${MRAY_LIB_INCLUDE_DIRECTORY}"
        # OptiX Documentation says that -G'ed kernels may fail
        # So -lineinfo is used on both configurations
        $<$<CONFIG:Debug>:-G>
        $<$<CONFIG:Release>:-lineinfo>
        #-lineinfo
        $<$<CONFIG:Debug>:-DMETU_DEBUG>
        $<$<CONFIG:Release>:-DNDEBUG>
        -DMETU_CUDA
        $<$<CONFIG:Release>:-DNDEBUG>
        #${ALL_COMPILE_DEFS}
     )

    # Custom Target Name
    set(PTX_TARGET "${NVCC_COMPILE_PTX_MAIN_TARGET}_Optix")

    # Custom build rule to generate ptx files from cuda files
    FOREACH(INPUT ${NVCC_COMPILE_PTX_SOURCES})

        get_filename_component(INPUT_STEM "${INPUT}" NAME_WE)

        # Generate New Ptx file for each CC Requested
        FOREACH(COMPUTE_CAPABILITY ${CMAKE_CUDA_ARCHITECTURES})
            # Generate the *.ptx files to the appropirate bin directory.
            set(OUTPUT_FILE "${INPUT_STEM}.CC_${COMPUTE_CAPABILITY}.o.ptx")
            set(OUTPUT_DIR "${MRAY_CONFIG_BIN_DIRECTORY}/OptiXShaders")
            set(OUTPUT "${OUTPUT_DIR}/${OUTPUT_FILE}")

            #set(DEP_FILE "${INPUT_STEM}.o.ptx.d")
            #set(DEP_DIR "${CMAKE_CURRENT_BINARY_DIR}/${NVCC_COMPILE_PTX_MAIN_TARGET}.dir/")
            #set(DEP_PATH "${DEP_DIR}/${DEP_FILE}")

            list(APPEND PTX_FILES ${OUTPUT})

            # This prints the standalone NVCC command line for each CUDA file.
            #message(STATUS "${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS} ${INPUT} -o ${OUTPUT} -odir ${OUTPUT_DIR}")
            #message(STATUS ${NVCC_COMPILE_OPTIONS})
            add_custom_command(
                OUTPUT  "${OUTPUT}"
                COMMENT "Builidng PTX File (CC_${COMPUTE_CAPABILITY}) ${INPUT}"
                DEPENDS "${INPUT}"
                # TODO: Check that if this works
                IMPLICIT_DEPENDS CXX "${INPUT}"
                # DEPFILE "${DEP_PATH}"
                # COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS}
                #         -M
                #         -o ${DEP_PATH}
                #         ${INPUT}
                #COMMAND ${CMAKE_CUDA_COMPILER} --help
                COMMAND ${CMAKE_CUDA_COMPILER} ${NVCC_COMPILE_OPTIONS}
                         "--gpu-architecture=compute_${COMPUTE_CAPABILITY}"
                         -o ${OUTPUT}
                         ${INPUT}
            )
        ENDFOREACH()
  ENDFOREACH()

  # Custom Target for PTX Files Main Target should depend on this target
  add_custom_target(${PTX_TARGET}
                    DEPENDS ${PTX_FILES}
                    # Add Source files for convenience
                    SOURCES ${NVCC_COMPILE_PTX_SOURCES})

  set(${NVCC_COMPILE_PTX_GENERATED_TARGET} ${PTX_TARGET} PARENT_SCOPE)
ENDFUNCTION()