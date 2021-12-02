cmake_minimum_required(VERSION 3.20)

# All CMake Related files is in the current folder
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

# Only Debug and Release
set(CMAKE_CONFIGURATION_TYPES Debug;Release CACHE STRING "Selected by user")
#set(CMAKE_DEFAULT_BUILD_TYPE Release CACHE STRING "Selected by user")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Selected by user")

# On Linux I dont know why but this is needs to be defined
# TODO: Maybe CUDA did not installed to default directory?
if(UNIX)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
endif()

# Architecture
# Use a dropdown menu for selecting Compute Capability
# TODO: Add a "ReleaseAllCC" Configuration for final release compilation
# including all compute capabilities etc.
set(CMAKE_CUDA_ARCHITECTURES "61" CACHE STRING "Architecture chosen by the user at CMake configure time")
set_property(CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY STRINGS 52 60 61 70 72 75 86 All)
if(${CMAKE_CUDA_ARCHITECTURES} STREQUAL "All")
    set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 72 75 86)
endif()

option(USE_OPTIX "Try to find and use OptiX" OFF)

# Globals Common for All Projects
# TODO: Align CUDA Standart with CXX Standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CUDA_RUNTIME_LIBRARY Shared)

# Find Optix If Available
if(USE_OPTIX)
    find_package(OPTIX 7)
    if(NOT OPTIX_FOUND)
        # Maybe we could not able to find the optix
        # expose the OPTIX_DIR (in cmake-gui) so user can manually set it
        message(FATAL_ERROR "Optix Not Found, either set OPTIX_INSTALL_DIR or disable USE_OPTIX")
        set(USE_OPTIX CACHE BOOL OFF)
    else()
        set(MRAY_PREPROCESSOR_DEFS_GENERIC ${MRAY_PREPROCESSOR_DEFS_GENERIC} -DMRAY_OPTIX)
    endif()
endif()

project(METURay LANGUAGES CUDA CXX)

# CMAKE Project Definitions and Source, Library Folders are seperate
# Directories
set(MRAY_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(MRAY_SOURCE_DIRECTORY ${MRAY_DIRECTORY}/Source)
set(MRAY_LIB_DIRECTORY ${MRAY_DIRECTORY}/Lib)
set(MRAY_LIB_INCLUDE_DIRECTORY ${MRAY_DIRECTORY}/Lib/Include)
set(MRAY_BIN_DIRECTORY ${MRAY_DIRECTORY}/Bin)
set(MRAY_RESOURCE_DIRECTORY ${MRAY_DIRECTORY}/Resources)
# Build is probably not used but its here for convenience
set(MRAY_BUILD_DIRECTORY ${MRAY_DIRECTORY}/Build)
# Working Dir is used for debugging (currently shaders are here so...)
# For Visual Studio Projects this is copied to WorkingDir property for executables
set(MRAY_WORKING_DIRECTORY ${MRAY_DIRECTORY}/WorkingDir)

# Determine Platform and Config
if(MSVC)
    set(MRAY_PLATFORM_NAME Win)
    set(MRAY_PREPROCESSOR_DEFS_GENERIC ${MRAY_PREPROCESSOR_DEFS_GENERIC}
        -D_UNICODE
        -DUNICODE
        -DNOMINMAX)
    # Filter ALL_BUILD and ZERO_CHECK projects
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
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else()
    # This should be set automatically I think...
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--compiler-bindir=${CMAKE_CXX_COMPILER}>)
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
                        $<$<COMPILE_LANGUAGE:CXX>:/Zi>
                        $<$<COMPILE_LANGUAGE:CXX>:/external:W0>
                        # Ignore inline functions are not defined warning
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4984>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4506>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zi>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/external:W0>)
                        # DROPPED W4 On MSVC it shows many uncessary info
                        # These Flags for W4 however /external does not work properly i think
                        # it still returns warnings and clutters the code extensively
                        # Thus these are commented out
                        # Disable "Padding Added" warning
                        # $<$<COMPILE_LANGUAGE:CXX>:/wd4324>
                        # $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4324>
                        # # '__cudaUnregisterBinaryUtil': unreferenced function with internal linkage has been removed
                        # $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4505>)
    add_link_options($<$<COMPILE_LANGUAGE:CXX>:/DEBUG>)
else()
    # Clang / GCC both use same syntax
    # Warnings
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
                        $<$<COMPILE_LANGUAGE:CXX>:-g3>
                        # Directly passing Wall etc. causes many warnings
                        # $<$<COMPILE_LANGUAGE:CUDA>:-Wreorder>
                        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-g3>)
endif()

# Cuda Compile Options
add_compile_options($<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>
                    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>)

# All kernels started to show this warning (when a kernel uses a virtual funcion)
# Supress that warning
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
# All precompiled libraries are resides here
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

# After Everything is set
# Include NVCC compile to ptx routine
# Since Optix programs require pre-compiled *.ptx files
if(USE_OPTIX)
    include("NVCC_compile_ptx")
endif()

# External Libs
# add_subdirectory(Ext)

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
