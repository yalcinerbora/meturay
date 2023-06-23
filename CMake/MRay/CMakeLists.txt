set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MRay)

source_group("" FILES ${CURRENT_SOURCE_DIR}/main.cpp)

add_executable(MRay ${CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(MRay PRIVATE
                      RayLib
                      mray::meta_compile_opts)

# Not-linked but dependent targets
add_dependencies(MRay
                 TracerCUDA
                 VisorGL)

# Debug Parameters (For my own use case)
if(MSVC)

    set(DEBUG_ARGS
        "-t TracerConfig.json -v VisorConfig.json -r 640x360 Scenes/modernPool.json")

    set_target_properties(MRay PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()