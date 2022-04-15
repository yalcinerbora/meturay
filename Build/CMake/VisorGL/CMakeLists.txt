set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/VisorGL)
set(SHADER_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/Shaders)
set(FONT_SOURCE_DIR ${MRAY_RESOURCE_DIRECTORY}/Fonts)

# FONTS
set(SRC_FONTS
    ${FONT_SOURCE_DIR}/VeraMono.ttf
    ${FONT_SOURCE_DIR}/icomoon.ttf)

# SOURCES
set(SRC_GL
    ${CURRENT_SOURCE_DIR}/GLConversionFunctions.h
    ${CURRENT_SOURCE_DIR}/ShaderGL.cpp
    ${CURRENT_SOURCE_DIR}/ShaderGL.h
    ${CURRENT_SOURCE_DIR}/TextureGL.cpp
    ${CURRENT_SOURCE_DIR}/TextureGL.h
    ${CURRENT_SOURCE_DIR}/ToneMapGL.cpp
    ${CURRENT_SOURCE_DIR}/ToneMapGL.h)

set(SRC_GUI
    ${CURRENT_SOURCE_DIR}/InfoPanel.cpp
    ${CURRENT_SOURCE_DIR}/InfoPanel.h
    ${CURRENT_SOURCE_DIR}/TMOptionWindow.cpp
    ${CURRENT_SOURCE_DIR}/TMOptionWindow.h
    ${CURRENT_SOURCE_DIR}/VisorGUI.cpp
    ${CURRENT_SOURCE_DIR}/VisorGUI.h
    ${CURRENT_SOURCE_DIR}/VisorGUIWindowI.h
    ${CURRENT_SOURCE_DIR}/IcoMoonFontTable.h
    ${CURRENT_SOURCE_DIR}/MainStatusBar.cpp
    ${CURRENT_SOURCE_DIR}/MainStatusBar.h
    ${CURRENT_SOURCE_DIR}/VisorWindowInput.cpp
    ${CURRENT_SOURCE_DIR}/VisorWindowInput.h)

set(SRC_SHADERS
    ${SHADER_SOURCE_DIR}/AccumInput.comp
    ${SHADER_SOURCE_DIR}/AvgDivisor.comp
    ${SHADER_SOURCE_DIR}/DTreeRender.frag
    ${SHADER_SOURCE_DIR}/DTreeRender.vert
    ${SHADER_SOURCE_DIR}/LumReduction.comp
    ${SHADER_SOURCE_DIR}/PGReferenceRender.comp
    ${SHADER_SOURCE_DIR}/PProcessGeneric.frag
    ${SHADER_SOURCE_DIR}/PProcessGeneric.vert
    ${SHADER_SOURCE_DIR}/TextureMaxReduction.comp
    ${SHADER_SOURCE_DIR}/TonemapAndGamma.comp
    ${SHADER_SOURCE_DIR}/STreeRender.comp
    ${SHADER_SOURCE_DIR}/AlphaBlend.comp)

set(GUIDE_DEBUG
    ${CURRENT_SOURCE_DIR}/GuideDebugGL.cpp
    ${CURRENT_SOURCE_DIR}/GuideDebugGL.h
    ${CURRENT_SOURCE_DIR}/GuideDebugGUI.cpp
    ${CURRENT_SOURCE_DIR}/GuideDebugGUI.h
    ${CURRENT_SOURCE_DIR}/GuideDebugGUIFuncs.h
    ${CURRENT_SOURCE_DIR}/GuideDebugStructs.h
    ${CURRENT_SOURCE_DIR}/GuideDebugTypeGen.h)

set(GUIDE_DEBUG_RENDERERS
    ${CURRENT_SOURCE_DIR}/GDebugRendererI.h
    ${CURRENT_SOURCE_DIR}/GDebugRendererPPG.cpp
    ${CURRENT_SOURCE_DIR}/GDebugRendererPPG.h
    ${CURRENT_SOURCE_DIR}/GDebugRendererRL.cpp
    ${CURRENT_SOURCE_DIR}/GDebugRendererRL.h
    ${CURRENT_SOURCE_DIR}/GDebugRendererReference.h
    ${CURRENT_SOURCE_DIR}/GDebugRendererReference.cpp)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/GLFWCallbackDelegator.h
    ${CURRENT_SOURCE_DIR}/GLFWCallbackDelegator.cpp
    ${CURRENT_SOURCE_DIR}/Structs.h
    ${CURRENT_SOURCE_DIR}/VisorGL.cpp
    ${CURRENT_SOURCE_DIR}/VisorGL.h
    ${CURRENT_SOURCE_DIR}/VisorGLEntry.cpp
    ${CURRENT_SOURCE_DIR}/VisorGLEntry.h)

set(SRC_ALL
    ${SRC_GL}
    ${SRC_GUI}
    ${GUIDE_DEBUG}
    ${GUIDE_DEBUG_RENDERERS}
    ${SRC_SHADERS}
    ${SRC_COMMON})

source_group("GL" FILES ${SRC_GL})
source_group("GUI" FILES ${SRC_GUI})
source_group("GuideDebug" FILES ${GUIDE_DEBUG})
source_group("GuideDebug/Debug Renderers" FILES ${GUIDE_DEBUG_RENDERERS})
source_group("Shaders" FILES ${SRC_SHADERS})
source_group("" FILES ${SRC_COMMON})

# Preprocessor Defs
set(PREPROCESSOR_GENERIC
    ${MRAY_PREPROCESSOR_DEFS_GENERIC}
    METU_SHARED_VISORGL)

# Platform Specific Libraries
# TBB for std::execution (clang & GCC)
if(MSVC)
    set(PLATFORM_SPEC_LIBRARIES opengl32 glew32 glfw3dll)
elseif(UNIX)
    set(PLATFORM_SPEC_LIBRARIES GL GLU GLEW glfw tbb)
endif()

# DLL
add_library(VisorGL SHARED ${SRC_ALL})

target_include_directories(VisorGL PRIVATE
                           ${MRAY_SOURCE_DIRECTORY})
target_include_directories(VisorGL SYSTEM PRIVATE
                           ${MRAY_LIB_INCLUDE_DIRECTORY})

target_compile_definitions(VisorGL PRIVATE ${PREPROCESSOR_GENERIC})

target_link_libraries(VisorGL
                      RayLib
                      ImageIO
                      Imgui
                      Imgui_tex_inspect
                      ${PLATFORM_SPEC_LIBRARIES})

# Copy Shaders to DLL Loc
add_custom_command(TARGET VisorGL POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory
                   "${MRAY_CONFIG_BIN_DIRECTORY}/Shaders"

                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${SRC_SHADERS}
                   "${MRAY_CONFIG_BIN_DIRECTORY}/Shaders")

# Copy Fonts
add_custom_command(TARGET VisorGL POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory
                   "${MRAY_CONFIG_BIN_DIRECTORY}/Fonts"

                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                   ${SRC_FONTS}
                   "${MRAY_CONFIG_BIN_DIRECTORY}/Fonts")

# # Additional Copy Operation for Shaders
add_custom_target(VisorGL_Shaders ALL
                  COMMAND ${CMAKE_COMMAND} -E make_directory
                  "${MRAY_CONFIG_BIN_DIRECTORY}/Shaders"

                  COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  ${SRC_SHADERS}
                  "${MRAY_CONFIG_BIN_DIRECTORY}/Shaders")
add_dependencies(VisorGL_Shaders VisorGL)

# Copy DLLS
if(MSVC)
    add_custom_command(TARGET VisorGL POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${MRAY_CONFIG_LIB_DIRECTORY}/glfw3.dll"
                       "${MRAY_CONFIG_LIB_DIRECTORY}/glew32.dll"
                       $<$<CONFIG:Debug>:"${MRAY_CONFIG_LIB_DIRECTORY}/glfw3.pdb">
                       $<$<CONFIG:Debug>:"${MRAY_CONFIG_LIB_DIRECTORY}/glew32.pdb">
                       $<$<CONFIG:Debug>:"${MRAY_CONFIG_LIB_DIRECTORY}/Imgui.pdb">
                       ${MRAY_CONFIG_BIN_DIRECTORY})
endif()