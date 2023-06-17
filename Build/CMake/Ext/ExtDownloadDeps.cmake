
include(ExternalProject)

# Cosmetic wrapper of "ExternalProject_Add"
#   All libraries will be written on to Lib/$(Platform)/$(Config)
#   folders. (Config is not mandatory but it is neat while debugging)
#
#   Header files will be on Lib/include

macro(mray_build_ext_dependency_git)

    # Parse Args
    set(oneValueArgs NAME URL TAG SOURCE_SUBDIR)
    set(multiValueArgs BUILD_ARGS DEPENDENCIES)
    cmake_parse_arguments(BUILD_SUBPROJECT "" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    set(SUBPROJECT_EXT_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/${BUILD_SUBPROJECT_NAME})
    set(SUBPROJECT_INSTALL_PATH ${MRAY_PLATFORM_LIB_DIRECTORY})
    set(SUBPROJECT_BUILD_PATH ${SUBPROJECT_EXT_DIR}/build)

    # if(NOT ${BUILD_SUBPROJECT_BUILD_ROOT} STREQUAL "")
    #     message(STATUS ${BUILD_SUBPROJECT_BUILD_ROOT})
    #     list(APPEND SUBPROJECT_CUSTOM_BUILD_STEP
    #             "${CMAKE_COMMAND} --build ./${BUILD_SUBPROJECT_BUILD_ROOT}")
    #     message(STATUS ${SUBPROJECT_CUSTOM_BUILD_STEP})
    # endif()

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

                # Common args (it will share the generator and compiler)
                LIST_SEPARATOR | # Use the alternate list separator
                CMAKE_ARGS
                    #-DCMAKE_BUILD_TYPE:STRING=$<CONFIG>
                    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                    -DCMAKE_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
                    -DCMAKE_GENERATOR_TOOLSET= ${CMAKE_GENERATOR_TOOLSET}

                    ${SUBPROJECT_CUSTOM_BUILD_STEP}

                    # Install Stuff
                    -DCMAKE_INSTALL_PREFIX:PATH=${MRAY_LIB_DIRECTORY}/
                    -DCMAKE_INSTALL_INCLUDEDIR=Include
                    -DCMAKE_INSTALL_DOCDIR=Docs/${BUILD_SUBPROJECT_NAME}
                    -DCMAKE_INSTALL_DATADIR=${MRAY_PLATFORM_NAME}/$<CONFIG>
                    -DCMAKE_INSTALL_LIBDIR=${MRAY_PLATFORM_NAME}/$<CONFIG>
                    -DCMAKE_INSTALL_BINDIR=${MRAY_PLATFORM_NAME}/$<CONFIG>

                    -DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}
                ${BUILD_SUBPROJECT_BUILD_ARGS}
                BUILD_COMMAND ${DEFAULT_BUILD_COMMAND}
                #BUILD_BYPRODUCTS ${CMAKE_INSTALL_PREFIX}/${CMAKE_STATIC_LIBRARY_PREFIX}timestwo${CMAKE_STATIC_LIBRARY_SUFFIX}
                BUILD_ALWAYS OFF
    )

    if(BUILD_SUBPROJECT_DEPENDENCIES)
        ExternalProject_Add_StepDependencies(${BUILD_SUBPROJECT_NAME}
                                             configure ${BUILD_SUBPROJECT_DEPENDENCIES})
    endif()

    # Instead of doing prefix path here just
    list(APPEND CMAKE_PREFIX_PATH ${SUBPROJECT_INSTALL_PATH})
    string(REPLACE ";" "|" CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
endmacro()