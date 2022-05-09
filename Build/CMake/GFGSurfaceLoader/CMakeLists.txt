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

# Preprocessor Defs
set(PREPROCESSOR_GENERIC
    ${MRAY_PREPROCESSOR_DEFS_GENERIC}
    METU_SHARED_GFG_LOADER)

# DLL
add_library(GFGSurfaceLoader SHARED ${SRC_ALL})

target_include_directories(GFGSurfaceLoader PRIVATE
                           ${MRAY_SOURCE_DIRECTORY})
target_include_directories(GFGSurfaceLoader SYSTEM PRIVATE
                           ${MRAY_LIB_INCLUDE_DIRECTORY})

target_compile_definitions(GFGSurfaceLoader PRIVATE ${PREPROCESSOR_GENERIC})

target_link_libraries(GFGSurfaceLoader
                      RayLib
                      GFGFileIO)