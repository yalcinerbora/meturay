

set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/ImageIO)

# SOURCES
set(SRC_ALL
    ${CURRENT_SOURCE_DIR}/ImageIO.h
    ${CURRENT_SOURCE_DIR}/ImageIO.cpp
    ${CURRENT_SOURCE_DIR}/ImageIOI.h
    ${CURRENT_SOURCE_DIR}/FreeImgRAII.h
    ${CURRENT_SOURCE_DIR}/EntryPoint.h
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp)

source_group("" FILES ${SRC_ALL})

# Preprocessor Defs
set(PREPROCESSOR_GENERIC
    ${MRAY_PREPROCESSOR_DEFS_GENERIC}
    METU_SHARED_IMAGE_IO)

# OpenEXR Library Header Inclusion (Imath)
set(MRAY_OPENEXR_INCLUDE_DIRECTORY ${MRAY_LIB_INCLUDE_DIRECTORY}/Imath)

# DLL
add_library(ImageIO SHARED ${SRC_ALL})

target_include_directories(ImageIO PRIVATE
                           ${MRAY_SOURCE_DIRECTORY})

target_include_directories(ImageIO SYSTEM PRIVATE
                           ${MRAY_OPENEXR_INCLUDE_DIRECTORY}
                           ${MRAY_LIB_INCLUDE_DIRECTORY})

target_compile_definitions(ImageIO PRIVATE ${PREPROCESSOR_GENERIC})

target_link_libraries(ImageIO PRIVATE
                      RayLib
                      OpenEXR
                      freeimage)

# Copy DLLS
if(MSVC)
    add_custom_command(TARGET ImageIO POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${MRAY_CONFIG_LIB_DIRECTORY}/FreeImage.dll"
                       "${MRAY_CONFIG_LIB_DIRECTORY}/OpenEXR.dll"
                       "${MRAY_CONFIG_LIB_DIRECTORY}/Imath.dll"
                       "${MRAY_CONFIG_LIB_DIRECTORY}/IlmThread.dll"
                       "${MRAY_CONFIG_LIB_DIRECTORY}/Iex.dll"
                       $<$<CONFIG:Debug>:"${MRAY_CONFIG_LIB_DIRECTORY}/FreeImage.pdb">
                       $<$<CONFIG:Debug>:"${MRAY_CONFIG_LIB_DIRECTORY}/OpenEXR.pdb">
                       ${MRAY_CONFIG_BIN_DIRECTORY})
endif()