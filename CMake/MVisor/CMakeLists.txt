

set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MVisor)

source_group("" FILES ${CURRENT_SOURCE_DIR}/main.cpp)

add_executable(MVisor ${CURRENT_SOURCE_DIR}/main.cpp)

target_link_libraries(MVisor PRIVATE
                      RayLib
                      mray::meta_compile_opts)

# Not-linked but dependent targets
add_dependencies(MVisor VisorGL)

# Debug Parameters
if(MSVC)

    set(DEBUG_ARGS
        "--gdb --gdbc=GuideDebug/GuideDebug.json")

    set_target_properties(MVisor PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()