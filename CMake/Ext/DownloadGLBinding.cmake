mray_build_ext_dependency_git(
    NAME glbinding_ext
    URL "https://github.com/cginternals/glbinding.git"
    TAG "aedb549941537a53c6ce2fed6848f6f27e7d42ad" # v.3.3.0
    SKIP_INSTALL
    LICENSE_NAME "LICENSE"
    BUILD_ARGS
        -DBUILD_SHARED_LIBS=ON
        -DOPTION_BUILD_EXAMPLES=OFF
        -DOPTION_BUILD_OWN_KHR_HEADERS=ON
        -DOPTION_BUILD_TOOLS=OFF
        # I am speculating this will be specifically usefull
        # since the functions are actual gl functions instead of func. ptrs
        -DOPTION_BUILD_WITH_LTO=ON
)

# glbinding does not let us to specify the install
# we will add custom build steps to copy the files to appropirate folders
# This is slightly better since glbindings have extereme amount of headers
# one for every GL version, we require the main gl one
set(GLBINDING_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/glbinding_ext)
set(GLBINDING_DIR_HEADER_COMMON ${GLBINDING_DIR}/src/glbinding_ext/source/glbinding/include/glbinding)
set(GLBINDING_DIR_KHR ${GLBINDING_DIR}/src/glbinding_ext/source/3rdparty/KHR/include/KHR)
set(GLBINDING_EXT_LIB_DIR ${GLBINDING_DIR}/build/$<CONFIG>)

# Install folders
set(GLBINDING_INSTALL_HEADER_DIR ${MRAY_LIB_INCLUDE_DIRECTORY}/glbinding/)
set(GLBINDING_INSTALL_LIB_DIR ${MRAY_CONFIG_LIB_DIRECTORY}/)
set(GLBINDING_INSTALL_KHR_DIR ${MRAY_LIB_INCLUDE_DIRECTORY}/glbinding/3rdparty/KHR)

# Generated-Headers
# Version header file
set(GLBINDING_VERSION_HEADER_FILE ${GLBINDING_DIR}/build/source/include/glbinding/glbinding-version.h)
# Other Generated Header Files
file(GLOB GLBINDING_GENERATED_HEADERS LIST_DIRECTORIES FALSE
        ${GLBINDING_DIR}/build/source/glbinding/include/glbinding/*.h)
# Non-generated headers
file(GLOB GLBINDING_HEADERS LIST_DIRECTORIES FALSE
         ${GLBINDING_DIR_HEADER_COMMON}/*.h)
file(GLOB GLBINDING_HEADERS_INL LIST_DIRECTORIES FALSE
         ${GLBINDING_DIR_HEADER_COMMON}/*.inl)

# DLLs (version dependent)
set(GLBINDING_DLL_FILE ${GLBINDING_EXT_LIB_DIR}/glbinding$<$<CONFIG:Debug>:d>${CMAKE_SHARED_LIBRARY_SUFFIX})
if(MSVC)
    set(GLBINDING_LIB_FILE ${GLBINDING_EXT_LIB_DIR}/glbinding$<$<CONFIG:Debug>:d>.lib)
endif()

# KHR Header
set(GLBINDING_KHR_FILE ${GLBINDING_DIR_KHR}/khrplatform.h)

# CMake exports

ExternalProject_Add_Step(glbinding_ext custom_install
                        COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
                            ${GLBINDING_DIR_HEADER_COMMON}/gl
                            ${GLBINDING_INSTALL_HEADER_DIR}/gl
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            ${GLBINDING_HEADERS}
                            ${GLBINDING_HEADERS_INL}
                            ${GLBINDING_VERSION_HEADER_FILE}
                            ${GLBINDING_GENERATED_HEADERS}
                            ${GLBINDING_INSTALL_HEADER_DIR}
                        COMMAND ${CMAKE_COMMAND} -E make_directory
                            ${GLBINDING_INSTALL_LIB_DIR}
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            ${GLBINDING_DLL_FILE}
                            ${GLBINDING_LIB_FILE}
                            ${GLBINDING_INSTALL_LIB_DIR}
                        COMMAND ${CMAKE_COMMAND} -E make_directory
                            ${GLBINDING_INSTALL_KHR_DIR}
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            ${GLBINDING_KHR_FILE}
                            ${GLBINDING_INSTALL_KHR_DIR}
                         COMMENT "Custom install(copy) step for glbindings"
                         DEPENDEES INSTALL
)
