

set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/AssimpSurfaceLoaders)

# SOURCES
set(SRC_ALL
    ${CURRENT_SOURCE_DIR}/AssimpForward.h
    ${CURRENT_SOURCE_DIR}/AssimpSurfaceLoaderPool.cpp
    ${CURRENT_SOURCE_DIR}/AssimpSurfaceLoaderPool.h
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp
    ${CURRENT_SOURCE_DIR}/EntryPoint.h
    ${CURRENT_SOURCE_DIR}/MetaSurfaceLoader.cpp
    ${CURRENT_SOURCE_DIR}/MetaSurfaceLoader.h)

source_group("" FILES ${SRC_ALL})

# DLL
add_library(AssimpSurfaceLoaders SHARED ${SRC_ALL})
# Preprocessor Defs extra
target_compile_definitions(AssimpSurfaceLoaders PRIVATE
                           METU_SHARED_ASSIMP_LOADER)

target_link_libraries(AssimpSurfaceLoaders PRIVATE
                      RayLib
                      assimp::assimp
                      mray::meta_compile_opts
)

# Copy DLLS
if(MSVC)
    add_custom_command(TARGET AssimpSurfaceLoaders POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${MRAY_CONFIG_LIB_DIRECTORY}/assimp$<$<CONFIG:Debug>:d>.dll"
                       # TODO: There is a bug on assimp isntall (o I am using it wrong)
                       # It does generate pdb, but install copy path is wrong
                       # comment it for now, (hopefully we would not need to step in to assimp while debugging)
                       #$<$<CONFIG:Debug>:"${MRAY_CONFIG_LIB_DIRECTORY}/assimp.pdb">
                       ${MRAY_CONFIG_BIN_DIRECTORY})
endif()