

add_executable(MRay ${MRAY_SOURCE_DIRECTORY}/MRay/main.cpp)

target_include_directories(MRay PUBLIC
                           ${MRAY_LIB_DIRECTORY}/include
                           ${MRAY_SOURCE_DIRECTORY})