

set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MVisor)

add_executable(MVisor ${CURRENT_SOURCE_DIR}/main.cpp)

source_group("" FILES ${CURRENT_SOURCE_DIR}/main.cpp)

target_include_directories(MVisor PRIVATE
                            ${MRAY_LIB_INCLUDE_DIRECTORY}
                            ${MRAY_SOURCE_DIRECTORY})

# Not-linked but dependent targets
add_dependencies(MVisor VisorGL)