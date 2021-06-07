
set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MTracer)

add_executable(MTracer ${CURRENT_SOURCE_DIR}/main.cpp)

source_group("" FILES ${CURRENT_SOURCE_DIR}/main.cpp)

target_include_directories(MTracer PRIVATE
                            ${MRAY_LIB_INCLUDE_DIRECTORY}
                            ${MRAY_SOURCE_DIRECTORY})