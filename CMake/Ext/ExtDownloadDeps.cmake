
include(ExternalProject)

# Cosmetic wrapper of "ExternalProject_Add"
#   All libraries will be written on to Lib/$(Platform)/$(Config)
#   folders. (Config is not mandatory but it is neat while debugging)
#
#   Header files will be on Lib/include
#
# It is written wrt. to
# https://github.com/jeffamstutz/superbuild_ospray/blob/main/macros.cmake

macro(append_cmake_prefix_path)
  list(APPEND CMAKE_PREFIX_PATH ${ARGN})
  string(REPLACE ";" "|" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
endmacro()

function(mray_build_ext_dependency_git)

    # Parse Args
    set(options SKIP_INSTALL)
    set(oneValueArgs NAME URL TAG SOURCE_SUBDIR OVERRIDE_INSTALL_PREFIX)
    set(multiValueArgs BUILD_ARGS DEPENDENCIES)

    cmake_parse_arguments(BUILD_SUBPROJECT "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    set(SUBPROJECT_EXT_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/${BUILD_SUBPROJECT_NAME})
    set(SUBPROJECT_BUILD_PATH ${SUBPROJECT_EXT_DIR}/build)

    if(BUILD_SUBPROJECT_SKIP_INSTALL)
        set(SUBPROJECT_INSTALL_COMMAND_ARG "INSTALL_COMMAND")
        # Could not make the empty string work, so printing a message
        # as a "install" command
        set(SUBPROJECT_INSTALL_COMMAND
            ${CMAKE_COMMAND} -E echo "\"Skipping default install for ${BUILD_SUBPROJECT_NAME}\"")
    endif()

    # Override install prefix if requested
    # this install will be intermediate install
    # after this function user can copy the files using the appropirate
    # location
    if(NOT "${BUILD_SUBPROJECT_OVERRIDE_INSTALL_PREFIX}" STREQUAL "")
        set(SUBPROJECT_INSTALL_PREFIX ${BUILD_SUBPROJECT_OVERRIDE_INSTALL_PREFIX}/)
    else()
        set(SUBPROJECT_INSTALL_PREFIX ${MRAY_LIB_DIRECTORY}/)
    endif()
    # Actual Call
    ExternalProject_Add(${BUILD_SUBPROJECT_NAME}
                PREFIX ${SUBPROJECT_EXT_DIR}
                BINARY_DIR ${SUBPROJECT_BUILD_PATH}
                BUILD_IN_SOURCE OFF

                # DL Repo
                GIT_REPOSITORY ${BUILD_SUBPROJECT_URL}
                GIT_TAG ${BUILD_SUBPROJECT_TAG}
                GIT_SHALLOW ON
                # Custom build root location if required
                SOURCE_SUBDIR ${BUILD_SUBPROJECT_SOURCE_SUBDIR}

                # In order to skip install
                # I could not get it to work with lists dunno why
                ${SUBPROJECT_INSTALL_COMMAND_ARG}
                ${SUBPROJECT_INSTALL_COMMAND}

                # Log the outputs instead of printing
                # except when there is an error
                LOG_DOWNLOAD OFF
                LOG_UPDATE ON
                LOG_PATCH ON
                LOG_CONFIGURE ON
                LOG_BUILD ON
                LOG_INSTALL OFF
                LOG_OUTPUT_ON_FAILURE ON

                # Common args (it will share the generator and compiler)
                LIST_SEPARATOR | # Use the alternate list separator
                CMAKE_ARGS
                    #-DCMAKE_BUILD_TYPE:STRING=$<CONFIG>
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
                    -DCMAKE_GENERATOR_TOOLSET= ${CMAKE_GENERATOR_TOOLSET}

                    # Install Stuff
                    -DCMAKE_INSTALL_PREFIX:PATH=${SUBPROJECT_INSTALL_PREFIX}
                    -DCMAKE_INSTALL_INCLUDEDIR=Include
                    -DCMAKE_INSTALL_DOCDIR=Docs/${BUILD_SUBPROJECT_NAME}
                    -DCMAKE_INSTALL_DATADIR=${MRAY_PLATFORM_NAME}/$<CONFIG>
                    -DCMAKE_INSTALL_LIBDIR=${MRAY_PLATFORM_NAME}/$<CONFIG>
                    -DCMAKE_INSTALL_BINDIR=${MRAY_PLATFORM_NAME}/$<CONFIG>

                    -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
                ${BUILD_SUBPROJECT_BUILD_ARGS}
                BUILD_ALWAYS OFF
    )

    if(BUILD_SUBPROJECT_DEPENDENCIES)
        ExternalProject_Add_StepDependencies(${BUILD_SUBPROJECT_NAME}
                                             configure ${BUILD_SUBPROJECT_DEPENDENCIES})
    endif()

endfunction()