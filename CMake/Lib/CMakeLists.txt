
# Configure Time check the availablity of the libraries
# that is build by "Ext" project
#
# TODO: Now this code is for development purposes
# On config time it builds both debug and release builds of the libraries
# However maybe for install etc user wants a single config of the MRay
# thus we only need to build that config for the external libraries too.
# Do an adjustment for that...

set(MRAY_DEBUG_LIB_DIRECTORY ${MRAY_PLATFORM_LIB_DIRECTORY}/Debug)
set(MRAY_RELEASE_LIB_DIRECTORY ${MRAY_PLATFORM_LIB_DIRECTORY}/Release)

set(MRAY_EXT_DIR ${MRAY_DIRECTORY}Ext/${MRAY_PLATFORM_NAME})
set(MRAY_CMAKE_EXT_DIR ${MRAY_DIRECTORY}/Cmake/Ext)

# This style is not good but it is fastest style
# since if we add these as a target etc.
# many steps are done by the CMake even it everything is up to date
# (5-10 seconds). This is tedious while developing
# Because of that it is done on the very first config.
# While building, cmake does not even check the external libs which is fast.
if(NOT EXISTS ${MRAY_DEBUG_LIB_DIRECTORY}/lib.stamp OR
   NOT EXISTS ${MRAY_RELEASE_LIB_DIRECTORY}/lib.stamp)

   message(NOTICE "Compiling external libraries. This may take a while...")

    set(MRAY_BUILD_DIR ${MRAY_EXT_DIR}/Build)
    set(MRAY_EXT_CONFIG_META_ARGS
        -S ${MRAY_CMAKE_EXT_DIR}
        -B ${MRAY_BUILD_DIR}
        -G ${CMAKE_GENERATOR}
        # Main Config Options
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DMRAY_BUILD_TESTS=${MRAY_BUILD_TESTS}
    )

    get_property(MRAY_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(MRAY_IS_MULTI_CONFIG)
        # We can configure once build twice
        # Configure the target
        execute_process(
            COMMAND ${CMAKE_COMMAND} ${MRAY_EXT_CONFIG_META_ARGS}
            COMMAND_ERROR_IS_FATAL ANY)
        # Build For Debug
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${MRAY_BUILD_DIR} --config Debug
            COMMAND_ERROR_IS_FATAL ANY)
        # Build For Release
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${MRAY_BUILD_DIR} --config Release
            COMMAND_ERROR_IS_FATAL ANY)

    else()
        # We can configure twice and build twice
        # Configure for Debug
        execute_process(
            COMMAND ${CMAKE_COMMAND} ${MRAY_EXT_CONFIG_META_ARGS} -DCMAKE_BUILD_TYPE=Debug
            COMMAND_ERROR_IS_FATAL ANY)
        # Build For Debug
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${MRAY_BUILD_DIR}
            COMMAND_ERROR_IS_FATAL ANY)
        # Configure for Release
        execute_process(
            COMMAND ${CMAKE_COMMAND} ${MRAY_EXT_CONFIG_META_ARGS} -DCMAKE_BUILD_TYPE=Release
            COMMAND_ERROR_IS_FATAL ANY)
        # Build For Release
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build ${MRAY_BUILD_DIR}
            COMMAND_ERROR_IS_FATAL ANY)

    endif()

else()

    message(STATUS "External libraries are compiled for both configurations")

endif()

# Everything should be fine now
# Our directory structure is solid
# I've tried find_package route but it is messy :(
# install config location is not standart for CMake
# or the standart is not compatible for multi-configuration
# structure of arrays style of directory layout
#
# We create imported targets by hand
# everything should be compiled properly
# Some include nuances exists but we will handle it here
#
# TODO: Change this maybe later?
function(gen_imported_target)
    # Parse Args
    set(options STATIC SHARED)
    set(oneValueArgs NAME DEBUG_TARGET RELEASE_TARGET)
    set(multiValueArgs EXTRA_INCLUDES)
    cmake_parse_arguments(GEN_IMPORTED_TGT "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})

    if(GEN_IMPORTED_TGT_STATIC AND GEN_IMPORTED_TGT_SHARED)
        message(FATAL_ERROR "gen_imported_target called with both SHARED and STATIC")
        return()
    endif()
    if(NOT GEN_IMPORTED_TGT_STATIC AND NOT GEN_IMPORTED_TGT_SHARED)
        message(FATAL_ERROR "gen_imported_target must be called with either SHARED or STATIC")
        return()
    endif()

    # Determine the naming stuff
    if(GEN_IMPORTED_TGT_STATIC)
        set(TGT_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
        set(TGT_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
    else()
        set(TGT_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(TGT_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
    endif()

    if(GEN_IMPORTED_TGT_STATIC)
        add_library(${GEN_IMPORTED_TGT_NAME} STATIC IMPORTED)
    else()
        add_library(${GEN_IMPORTED_TGT_NAME} SHARED IMPORTED)
    endif()
    set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES IMPORTED_GLOBAL TRUE)


    set(DBG_NAME "${TGT_PREFIX}${GEN_IMPORTED_TGT_DEBUG_TARGET}")
    set(REL_NAME "${TGT_PREFIX}${GEN_IMPORTED_TGT_RELEASE_TARGET}")

    # message(NOTICE "${DBG_NAME}   ${REL_NAME}")

    set_property(TARGET ${GEN_IMPORTED_TGT_NAME} APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
    set_property(TARGET ${GEN_IMPORTED_TGT_NAME} APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)

    set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES
        IMPORTED_LOCATION_DEBUG "${MRAY_PLATFORM_LIB_DIRECTORY}/Debug/${DBG_NAME}${TGT_SUFFIX}"
        IMPORTED_LOCATION_RELEASE "${MRAY_PLATFORM_LIB_DIRECTORY}/Release/${REL_NAME}${TGT_SUFFIX}"
    )

    if(GEN_IMPORTED_TGT_EXTRA_INCLUDES)
        target_include_directories(${GEN_IMPORTED_TGT_NAME}
                                    INTERFACE
                                    ${GEN_IMPORTED_TGT_EXTRA_INCLUDES})
    endif()

    if(MSVC AND NOT GEN_IMPORTED_TGT_STATIC)
        set_target_properties(${GEN_IMPORTED_TGT_NAME} PROPERTIES
                              IMPORTED_IMPLIB_DEBUG "${MRAY_PLATFORM_LIB_DIRECTORY}/Debug/${DBG_NAME}.lib"
                              IMPORTED_IMPLIB_RELEASE "${MRAY_PLATFORM_LIB_DIRECTORY}/Release/${REL_NAME}.lib")
    endif()

endfunction()

gen_imported_target(NAME ZLIB::ZLIB SHARED DEBUG_TARGET ZLIBd RELEASE_TARGET ZLIBd)
gen_imported_target(NAME fmt::fmt STATIC DEBUG_TARGET fmt RELEASE_TARGET fmt)
gen_imported_target(NAME assimp::assimp SHARED DEBUG_TARGET assimpd RELEASE_TARGET assimp)
gen_imported_target(NAME GFGFileIO::GFGFileIO STATIC DEBUG_TARGET GFGFileIOd RELEASE_TARGET GFGFileIO)
gen_imported_target(NAME spdlog::spdlog SHARED DEBUG_TARGET spdlogd RELEASE_TARGET spdlog)
gen_imported_target(NAME glbinding::glbinding SHARED DEBUG_TARGET glbindingd RELEASE_TARGET glbinding
                    EXTRA_INCLUDES ${MRAY_LIB_INCLUDE_DIRECTORY}/glbinding/3rdparty)
gen_imported_target(NAME glfw3::glfw3 SHARED DEBUG_TARGET glfw3ddll RELEASE_TARGET glfw3dll)
gen_imported_target(NAME imgui::imgui STATIC DEBUG_TARGET imguid RELEASE_TARGET imgui)
gen_imported_target(NAME imgui::imgui_ti STATIC DEBUG_TARGET imgui_tex_inspectd RELEASE_TARGET imgui_tex_inspect)
gen_imported_target(NAME oiio::oiio SHARED DEBUG_TARGET OpenImageIOd RELEASE_TARGET OpenImageIO)
gen_imported_target(NAME oiio::oiio_util SHARED DEBUG_TARGET OpenImageIO_Utild RELEASE_TARGET OpenImageIO_Util)

if(MRAY_BUILD_TESTS)
    gen_imported_target(NAME gtest::gtest SHARED DEBUG_TARGET gtest RELEASE_TARGET gtest)
    gen_imported_target(NAME gtest::gmock SHARED DEBUG_TARGET gmock RELEASE_TARGET gmock)
endif()


# Generate copy dll target
# usefull for developing

# Now we can find packages
# find_package(ZLIB REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(ZLIB_DIR)
# find_package(fmt REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(fmt_DIR)
# find_package(nlohmann_json REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(nlohmann_json_DIR)
# find_package(assimp REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(assimp_DIR)
# find_package(GFGFileIO REQUIRED CONFIG  NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(GFGFileIO_DIR)
# find_package(spdlog REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(spdlog_DIR)
# find_package(KHRplatform REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(KHRplatform_DIR)
# find_package(glbinding REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(glbinding_DIR)
# find_package(glfw3 REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(glfw3_DIR)
# find_package(CLI11 REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(CLI11_DIR)
# find_package(imgui REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(imgui_DIR)
# find_package(imgui_tex_inspect REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
# mark_as_advanced(imgui_tex_inspect_DIR)
# if(MRAY_BUILD_TESTS)
#     find_package(GTest REQUIRED CONFIG NO_SYSTEM_ENVIRONMENT_PATH)
#     mark_as_advanced(GTest_DIR)
# endif()
