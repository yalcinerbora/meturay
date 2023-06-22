
set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MTracer)

add_executable(MTracer ${CURRENT_SOURCE_DIR}/main.cpp)

source_group("" FILES ${CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(MTracer PRIVATE
                      mray::meta_compile_opts)

# Not-linked but dependent targets
add_dependencies(MRay cli11_ext)