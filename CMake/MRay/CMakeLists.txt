

set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MRay)

source_group("" FILES ${CURRENT_SOURCE_DIR}/main.cpp)

add_executable(MRay ${CURRENT_SOURCE_DIR}/main.cpp)

add_dependencies(MRay cli11_ext)

target_include_directories(MRay PRIVATE
                           ${MRAY_SOURCE_DIRECTORY})

target_include_directories(MRay SYSTEM PRIVATE
                           ${MRAY_LIB_INCLUDE_DIRECTORY})

target_compile_definitions(MRay PRIVATE ${MRAY_PREPROCESSOR_DEFS_GENERIC})

target_link_libraries(MRay PRIVATE
                      RayLib)

# Not-linked but dependent targets
add_dependencies(MRay
                 TracerCUDA
                 VisorGL)

# Debug Parameters
if(MSVC)

    set(DEBUG_ARGS
        "-t TracerConfig.json -v VisorConfig.json -r 640x360 Scenes/modernPool.json")

    set_target_properties(MRay PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()