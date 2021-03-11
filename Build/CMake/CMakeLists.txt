cmake_minimum_required(VERSION 3.20)


project(meturay LANGUAGES CUDA)

#find_package(CUDA)
#set(CMAKE_CUDA_STANDARD 14)

# Globals
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(MTracer ../../Source/MTracer/main.cxx)

#set_target_properties(
#        cuda_test
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)