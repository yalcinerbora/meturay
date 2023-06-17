
set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MTracer)

add_executable(MTracer ${CURRENT_SOURCE_DIR}/main.cpp)

add_dependencies(MTracer cl11_ext)

source_group("" FILES ${CURRENT_SOURCE_DIR}/main.cpp)

target_include_directories(MTracer PRIVATE
                           ${MRAY_SOURCE_DIRECTORY})

target_include_directories(MTracer SYSTEM PRIVATE
                           ${MRAY_LIB_INCLUDE_DIRECTORY})