set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/GFGSurfaceLoader)

# SOURCES
set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp
    ${CURRENT_SOURCE_DIR}/EntryPoint.h
    ${CURRENT_SOURCE_DIR}/GFGSurfaceLoader.cpp
    ${CURRENT_SOURCE_DIR}/GFGSurfaceLoader.h
    ${CURRENT_SOURCE_DIR}/GFGSurfaceLoaderPool.cpp
    ${CURRENT_SOURCE_DIR}/GFGSurfaceLoaderPool.h)

set(SRC_ALL
    ${SRC_COMMON})

source_group("" FILES ${SRC_COMMON})

# DLL
add_library(GFGSurfaceLoader SHARED ${SRC_ALL})
add_dependencies(GFGSurfaceLoader gfg_ext)

# Preprocessor Defs
target_compile_definitions(GFGSurfaceLoader PRIVATE
                           METU_SHARED_GFG_LOADER)

target_link_libraries(GFGSurfaceLoader PRIVATE
                      RayLib
                      GFGFileIO
                      mray::meta_compile_opts)