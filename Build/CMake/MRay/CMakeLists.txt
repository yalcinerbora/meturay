

set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MRay)

source_group("" FILES ${CURRENT_SOURCE_DIR}/main.cpp)

add_executable(MRay ${CURRENT_SOURCE_DIR}/main.cpp)

target_include_directories(MRay PRIVATE
                           ${MRAY_LIB_INCLUDE_DIRECTORY}
                           ${MRAY_SOURCE_DIRECTORY})

target_compile_definitions(MRay PRIVATE ${MRAY_PREPROCESSOR_DEFS_GENERIC})

target_link_libraries(MRay PRIVATE
                      RayLib)

if(MSVC)

    set(DEBUG_ARGS
        -t TracerConfig.json
        -v VisorConfig.json
        -r  640x360
        Scenes/modernPool.json)

    set_target_properties(MRay PROPERTIES
                          VS_DEBUGGER_WORKING_DIRECTORY ${MRAY_WORKING_DIRECTORY}
                          VS_DEBUGGER_COMMAND_ARGUMENTS ${DEBUG_ARGS})
endif()