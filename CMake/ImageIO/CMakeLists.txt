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

# DLL
add_library(ImageIO SHARED ${SRC_ALL})

target_compile_definitions(ImageIO PRIVATE
                           METU_SHARED_IMAGE_IO)

target_link_libraries(ImageIO PRIVATE
                      RayLib
                      OpenEXR
                      #oiio
                      mray::meta_compile_opts)

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