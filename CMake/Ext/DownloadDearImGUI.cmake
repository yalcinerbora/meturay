

mray_build_ext_dependency_git(
    NAME imgui_ext
    URL "https://github.com/ocornut/imgui.git"
    TAG "5319d1cffafd5045c4742892c38c9e5cfa23d195" # v1.89.6
    LICENSE_NAME "LICENSE.txt"
    BUILD_ARGS
        -DCMAKE_MODULE_PATH=${MRAY_CONFIG_LIB_DIRECTORY}/cmake
    DEPENDS
        glfw_ext

)

set(MRAY_IMGUI_SRC_LOCATION ${MRAY_PLATFORM_EXT_DIRECTORY}/imgui_ext/src/imgui_ext)

# Inject an custom cmakelists.txt and make cmake to use it
ExternalProject_Add_Step(imgui_ext inject_cmake
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists-DearImGUI.cmake
                        ${MRAY_IMGUI_SRC_LOCATION}/CMakeLists.txt
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${CMAKE_CURRENT_SOURCE_DIR}/imguiConfig.cmake.in
                        ${MRAY_IMGUI_SRC_LOCATION}/imguiConfig.cmake.in
                DEPENDEES DOWNLOAD UPDATE PATCH
                DEPENDERS CONFIGURE
                COMMENT "Injecting a cmake lists to dearimgui"
)