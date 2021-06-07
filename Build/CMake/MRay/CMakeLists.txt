

set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/MRay)

source_group("" FILES ${CURRENT_SOURCE_DIR}/main.cpp)

add_executable(MRay ${CURRENT_SOURCE_DIR}/main.cpp)

target_include_directories(MRay PRIVATE
                           ${MRAY_LIB_INCLUDE_DIRECTORY}
                           ${MRAY_SOURCE_DIRECTORY})

target_compile_definitions(MRay PRIVATE ${MRAY_PREPROCESSOR_DEFS_GENERIC})

target_link_libraries(MRay PRIVATE
                      RayLib)