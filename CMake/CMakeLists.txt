cmake_minimum_required(VERSION 3.26)

# All CMake Related files is in the current folder
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

# Do not use in source builds
include(PreventInSourceBuilds)

# Only Debug and Release
set(CMAKE_CONFIGURATION_TYPES Debug;Release CACHE STRING "Selected by user")
#set(CMAKE_DEFAULT_BUILD_TYPE Release CACHE STRING "Selected by user")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Selected by user")

# On Linux I don't know why but this is needs to be defined
# Maybe CUDA did not installed to default directory?
if(UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# Architecture
# Use a drop-down menu for selecting Compute Capability
set(CMAKE_CUDA_ARCHITECTURES "native"
    CACHE STRING "Architecture chosen by the user at CMake configure time")
set_property(CACHE CMAKE_CUDA_ARCHITECTURES
             PROPERTY STRINGS native 52 60 61 70 72 75 86 89 90 all all-major)

option(MRAY_USE_OPTIX "Try to find and use OptiX" OFF)
option(MRAY_BUILD_TESTS "Build Tests" ON)

# Globals Common for All Projects
# TODO: Align CUDA Standard with CXX Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Find Optix If Available
if(MRAY_USE_OPTIX)
    find_package(OPTIX 7)
    if(NOT OPTIX_FOUND)
        # Maybe we could not able to find the optix
        # expose the OPTIX_DIR (in cmake-GUI) so user can manually set it
        message(FATAL_ERROR "Optix Not Found, either set OPTIX_INSTALL_DIR or disable MRAY_USE_OPTIX")
        set(MRAY_USE_OPTIX CACHE BOOL OFF)
    endif()
endif()

# Project Name etc..
project(METURay
        LANGUAGES CUDA CXX
        VERSION 0.1.0
        DESCRIPTION "Research oriented renderer."
)

include(Macros)
# CMAKE Project Definitions and Source, Library Folders are separate
# Directories
set(MRAY_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
cmake_path(NORMAL_PATH MRAY_DIRECTORY)
# Populate the meta directories for this project hierarchy
generate_platform_name()
populate_mray_directory_variables(${MRAY_DIRECTORY})

# After project is set and nvcc is found invoke find CUDA toolkit
# we will use curand for sampling
find_package(CUDAToolkit REQUIRED)
# Bug on cmake maybe this variable is not advanced
mark_as_advanced(CUDAToolkit_CUPTI_INCLUDE_DIR)

# Determine Platform and Config
# Cosmetic stuff
if(MSVC)
    # Filter ALL_BUILD and ZERO_CHECK projects
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    set(PREDEFINED_TARGETS_FOLDER "CustomTargets")
elseif(UNIX AND NOT APPLE)
    set(MRAY_PLATFORM_NAME Linux)
else()
    message(FATAL_ERROR "Unknown platform... Terminating CMake.")
    return()
endif()

# Use currently selected compiler for cude
# This is not set?
if(MSVC)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else()
    # This should be set automatically I think...
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--compiler-bindir=${CMAKE_CXX_COMPILER}>)
endif()

# Force Colored Output on Ninja
set(CMAKE_COLOR_DIAGNOSTICS ON)

# Compile Option Targets
# Generates
#   mray::meta_compile_opts
#   mray::cuda_extra_compile_opts
include(CompilerOptions.cmake)

# After Everything is set
# Include NVCC compile to ptx routine
# Since Optix programs require precompiled *.ptx files
if(MRAY_USE_OPTIX)
    include("NVCCCompilePTX")
endif()

# Enable LTO / Device-LTO
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE FASLE)
#set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG FALSE)

# External Libs
add_subdirectory(Lib)

# BUILDS
# Libs
add_subdirectory(RayLib)

# DLLs
add_subdirectory(AssimpSurfaceLoaders)
add_subdirectory(GFGSurfaceLoader)
add_subdirectory(ImageIO)
add_subdirectory(TracerCUDA)
add_subdirectory(VisorGL)

# Execs
add_subdirectory(MRay)
add_subdirectory(MVisor)
#add_subdirectory(MTracer)

# Tests
if(MRAY_BUILD_TESTS)
    add_subdirectory(Test_TracerCUDA)
    add_subdirectory(Test_Misc)
endif()
