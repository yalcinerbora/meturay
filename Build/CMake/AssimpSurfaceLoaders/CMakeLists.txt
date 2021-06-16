

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

# Preprocessor Defs
set(PREPROCESSOR_GENERIC
    ${MRAY_PREPROCESSOR_DEFS_GENERIC}
    METU_SHARED_ASSIMP_LOADER)

# DLL
add_library(AssimpSurfaceLoaders SHARED ${SRC_ALL})

target_include_directories(AssimpSurfaceLoaders PRIVATE
                           ${MRAY_LIB_INCLUDE_DIRECTORY}
                           ${MRAY_SOURCE_DIRECTORY})

target_compile_definitions(AssimpSurfaceLoaders PRIVATE ${PREPROCESSOR_GENERIC})

target_link_libraries(AssimpSurfaceLoaders PRIVATE
                      RayLib
                      assimp)



#add_dependencies(AssimpSurfaceLoaders RayLib)