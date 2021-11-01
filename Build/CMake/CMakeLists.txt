cmake_minimum_required(VERSION 3.20)

# Only Debug and Release
set(CMAKE_CONFIGURATION_TYPES Debug;Release CACHE STRING "Selected by user")
#set(CMAKE_DEFAULT_BUILD_TYPE Release CACHE STRING "Selected by user")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Selected by user")

if(UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# Architecture
set(CMAKE_CUDA_ARCHITECTURES "61" CACHE STRING "Architecture chosen by the user at CMake configure time")
set_property(CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY STRINGS 52 60 61 70 72 75 86)

# Globals
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CUDA_RUNTIME_LIBRARY Shared)

project(METURay LANGUAGES CUDA CXX)

# Some User Properties

# Directories
set(MRAY_SOURCE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../Source)
set(MRAY_LIB_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../Lib)
set(MRAY_LIB_INCLUDE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../Lib/Include)
set(MRAY_BIN_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../Bin)
# Build is not used but its here for convenience
set(MRAY_BUILD_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../Build)
# Working Dir is used for debugging (currently shaders are here so...)
set(MRAY_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../WorkingDir)

# Determine Platform and Config
if(MSVC)
    set(MRAY_PLATFORM_NAME Win)
    set(MRAY_PREPROCESSOR_DEFS_GENERIC -D_UNICODE -DUNICODE)
    # Ignore inline function not defined warning
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4984>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4506>)
    # Filter annoying ALL_BUILD ZERO_CHECK projects
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    set(PREDEFINED_TARGETS_FOLDER "CustomTargets")
elseif(APPLE)
    set(MRAY_PLATFORM_NAME Mac)
elseif(UNIX)
    set(MRAY_PLATFORM_NAME Linux)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
    message(FATAL_ERROR "Unknown platform... Terminating CMake.")
endif()

# Use currently selected compiler
if(MSVC)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_C_COMPILER})
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--compiler-bindir=${CMAKE_C_COMPILER}>)
endif()
# Force Colored Output on Ninja
if ((CMAKE_GENERATOR MATCHES "Ninja") AND UNIX)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        add_compile_options(-fcolor-diagnostics)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_compile_options(-fdiagnostics-color=always)
    endif()
endif()

# Generic Compile Options
if(MSVC)
    # Warnings
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W3>
                        $<$<COMPILE_LANGUAGE:CXX>:/Zi>)
    add_link_options($<$<COMPILE_LANGUAGE:CXX>:/DEBUG>)
else()
    # Clang / GCC both use same syntax
    # Warnings
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                        $<$<COMPILE_LANGUAGE:CXX>:-g3>)
endif()

# Cuda Compile Options
add_compile_options($<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
                    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>)

# All kernels started to show this warning (if a kernel uses a virtual funcion)
# Supress that warning
#add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--nvlink-options=-suppress-stack-size-warning>)
#add_link_options($<$<COMPILE_LANGUAGE:CUDA>:>)
add_link_options($<DEVICE_LINK:--nvlink-options=-suppress-stack-size-warning>)

# Platform Specific Lib Bin Folders
set(MRAY_PLATFORM_LIB_DIRECTORY ${MRAY_LIB_DIRECTORY}/${MRAY_PLATFORM_NAME})
set(MRAY_PLATFORM_BIN_DIRECTORY ${MRAY_BIN_DIRECTORY}/${MRAY_PLATFORM_NAME})
# Platform & Configurations Related Lib Bin folders
set(MRAY_CONFIG_LIB_DIRECTORY ${MRAY_PLATFORM_LIB_DIRECTORY}/$<CONFIG>)
set(MRAY_CONFIG_BIN_DIRECTORY ${MRAY_PLATFORM_BIN_DIRECTORY}/$<CONFIG>)
# Set cmake vars for output
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${MRAY_CONFIG_BIN_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${MRAY_CONFIG_BIN_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MRAY_CONFIG_BIN_DIRECTORY})
# Meta library directory
link_directories(${MRAY_CONFIG_LIB_DIRECTORY}
                 ${MRAY_PLATFORM_LIB_DIRECTORY})

# Generic Preprocessor Defitions
set(MRAY_PREPROCESSOR_DEFS_GENERIC
    ${MRAY_PREPROCESSOR_DEFS_GENERIC}
    $<$<CONFIG:Debug>:METU_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>)

set(MRAY_PREPROCESSOR_DEFS_CUDA_GENERIC
    ${MRAY_PREPROCESSOR_DEFS_GENERIC}
    METU_CUDA)

# BUILDS
# Libs
add_subdirectory(RayLib)

# DLLs
add_subdirectory(AssimpSurfaceLoaders)
add_subdirectory(ImageIO)
add_subdirectory(TracerCUDA)
add_subdirectory(VisorGL)

# Execs
add_subdirectory(MRay)
add_subdirectory(MVisor)
#add_subdirectory(MTracer)

# Tests
add_subdirectory(Test_TracerCUDA)
add_subdirectory(Test_Misc)
