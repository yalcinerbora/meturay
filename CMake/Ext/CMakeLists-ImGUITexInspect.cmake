cmake_minimum_required(VERSION 3.25)

include(GNUInstallDirs)

project(ImguiTexInspectInject LANGUAGES CXX)

find_package(imgui REQUIRED)
#find_package(glbinding REQUIRED)

add_library(imgui_tex_inspect STATIC)

# TODO: Can we neatly install this?
# An idea inject a CMakeLists.txt
# to dearimgui (add as a before configure step thing)
# then compile and run

target_sources(imgui_tex_inspect PUBLIC FILE_SET HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} # This is correct?
            FILES
            # Headers
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui_tex_inspect_internal.h
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui_tex_inspect.h
)

target_sources(imgui_tex_inspect PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui_tex_inspect.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/imgui_tex_inspect_demo.cpp

            ${CMAKE_CURRENT_SOURCE_DIR}/backends/tex_inspect_opengl.cpp
)

# imgui backends directly include imgui.h
# so we cannot add backends to backend folder directly install these
set(IMGUITI_BACKEND_HEADERS
    # Backend Headers
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/tex_inspect_opengl.h
)

target_link_libraries(imgui_tex_inspect imgui)

target_compile_definitions(imgui_tex_inspect PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)

target_include_directories(imgui_tex_inspect PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../Lib/Include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../Lib/Include/glbinding/3rdparty)

install(TARGETS imgui_tex_inspect
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        FILE_SET HEADERS
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Imgui)

install(FILES  ${IMGUITI_BACKEND_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/Imgui)
