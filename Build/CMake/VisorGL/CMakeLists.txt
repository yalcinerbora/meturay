set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/VisorGL)
set(SHADER_SOURCE_DIR ${MRAY_WORKING_DIRECTORY}/Shaders)

# SOURCES
set(SRC_GL
    ${CURRENT_SOURCE_DIR}/GLConversionFunctions.h
    ${CURRENT_SOURCE_DIR}/ShaderGL.cpp
    ${CURRENT_SOURCE_DIR}/ShaderGL.h
    ${CURRENT_SOURCE_DIR}/ToneMapGL.cpp
    ${CURRENT_SOURCE_DIR}/ToneMapGL.h)

set(SRC_GUI
    ${CURRENT_SOURCE_DIR}/TMOptionWindow.cpp
    ${CURRENT_SOURCE_DIR}/TMOptionWindow.h
    ${CURRENT_SOURCE_DIR}/VisorGUI.cpp
    ${CURRENT_SOURCE_DIR}/VisorGUI.h
    ${CURRENT_SOURCE_DIR}/VisorGUIWindowI.h)

set(SRC_SHADERS
    ${SHADER_SOURCE_DIR}/AccumInput.comp
    ${SHADER_SOURCE_DIR}/AvgDivisor.comp
    ${SHADER_SOURCE_DIR}/LumReduction.comp
    ${SHADER_SOURCE_DIR}/PProcessGeneric.frag
    ${SHADER_SOURCE_DIR}/PProcessGeneric.vert
    ${SHADER_SOURCE_DIR}/TonemapAndGamma.comp)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/Structs.h
    ${CURRENT_SOURCE_DIR}/VisorGL.cpp
    ${CURRENT_SOURCE_DIR}/VisorGL.h
    ${CURRENT_SOURCE_DIR}/VisorGLEntry.cpp
    ${CURRENT_SOURCE_DIR}/VisorGLEntry.h)

set(SRC_ALL    
    ${SRC_GL}
    ${SRC_GUI}
    ${SRC_SHADERS}
    ${SRC_COMMON})

source_group("GL" FILES ${SRC_GL})
source_group("GUI" FILES ${SRC_GUI})
source_group("Shaders" FILES ${SRC_SHADERS})
source_group("" FILES ${SRC_COMMON})

# Preprocessor Defs
set(PREPROCESSOR_GENERIC 
    ${MRAY_PREPROCESSOR_DEFS_GENERIC}
    METU_SHARED_VISORGL)

# DLL
add_library(VisorGL SHARED ${SRC_ALL})

target_include_directories(VisorGL PRIVATE
                           ${MRAY_LIB_DIRECTORY}/include
                           ${MRAY_SOURCE_DIRECTORY})

target_compile_definitions(VisorGL PRIVATE ${PREPROCESSOR_GENERIC})

target_link_libraries(VisorGL 
                      RayLib
                      Imgui
                      glew32
                      glfw3dll
                      opengl32)