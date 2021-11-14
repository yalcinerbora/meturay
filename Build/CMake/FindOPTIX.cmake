
# Using this tutorial to implement this
# https://gitlab.kitware.com/cmake/community/-/wikis/doc/tutorials/How-To-Find-Libraries
#
# and looking to here
# https://github.com/NVIDIA/OptiX_Apps
# and lookig in the optix samples

# Check if Optix Path is coming from command line
set(OPTIX_INSTALL_DIR $ENV{OPTIX_INSTALL_DIR})

# If not try to find the Optix

if(WIN32)
    # Default Installation Locations
    set(OPTIX_POTENTIAL_PATH_LIST
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0"
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0"
        )
else()
    set(OPTIX_POTENTIAL_PATH_LIST
            "~/NVIDIA-OptiX-SDK-7.5.0-linux64"
            "~/NVIDIA-OptiX-SDK-7.4.0-linux64"
            "~/NVIDIA-OptiX-SDK-7.3.0-linux64"
            "~/NVIDIA-OptiX-SDK-7.2.0-linux64"
            "~/NVIDIA-OptiX-SDK-7.1.0-linux64"
            "~/NVIDIA-OptiX-SDK-7.0.0-linux64"
        )
endif()

# Try to find optix header
if("${OPTIX_INSTALL_DIR}" STREQUAL "")
    find_path(OPTIX_INSTALL_DIR
        NAME include/optix.h
        PATHS ${OPTIX_POTENTIAL_PATH_LIST}
    )
endif()

# Set Include Folder
set(OPTIX_INCLUDE_DIR ${OPTIX_INSTALL_DIR}/include)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(OPTIX DEFAULT_MSG OPTIX_INCLUDE_DIR)
