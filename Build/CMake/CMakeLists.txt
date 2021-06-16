cmake_minimum_required(VERSION 3.20)

# Only Debug and Release
set(CMAKE_CONFIGURATION_TYPES Debug;Release CACHE STRING "Selected by user")
#set(CMAKE_DEFAULT_BUILD_TYPE Release CACHE STRING "Selected by user")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Selected by user")

if(UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# Architecture
set(CMAKE_CUDA_ARCHITECTURES "52" CACHE STRING "Architecture chosen by the user at CMake configure time")
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

# Generic compile options

# Libs
add_subdirectory(RayLib)

# DLLs
add_subdirectory(AssimpSurfaceLoaders)
add_subdirectory(TracerCUDA)
add_subdirectory(VisorGL)

# Execs
add_subdirectory(MRay)
#add_subdirectory(MTracer)
#add_subdirectory(MVisor)

# Tests
add_subdirectory(Test_TracerCUDA)
add_subdirectory(Test_Misc)
