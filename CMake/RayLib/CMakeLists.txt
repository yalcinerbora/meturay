set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/RayLib)

# SOURCES
set(SRC_CPU_CONTAINERS
    ${CURRENT_SOURCE_DIR}/MPMCQueue.h
    ${CURRENT_SOURCE_DIR}/MPMCQueue.hpp
    ${CURRENT_SOURCE_DIR}/StaticString.h)

set(SRC_DISTRIBUTOR
    ${CURRENT_SOURCE_DIR}/DistributorI.h
    ${CURRENT_SOURCE_DIR}/MDistributor.cpp
    ${CURRENT_SOURCE_DIR}/MDistributor.h)

set(SRC_ERRORS
    ${CURRENT_SOURCE_DIR}/DLLError.h
    ${CURRENT_SOURCE_DIR}/Error.h
    ${CURRENT_SOURCE_DIR}/IOError.h
    ${CURRENT_SOURCE_DIR}/NodeError.h
    ${CURRENT_SOURCE_DIR}/SceneError.h
    ${CURRENT_SOURCE_DIR}/TracerError.h
    ${CURRENT_SOURCE_DIR}/VisorError.h
    ${CURRENT_SOURCE_DIR}/ImageIOError.h)

set(SRC_FILE_IO
    ${CURRENT_SOURCE_DIR}/ConfigParsers.cpp
    ${CURRENT_SOURCE_DIR}/ConfigParsers.h
    ${CURRENT_SOURCE_DIR}/EnumStringConversions.cpp
    ${CURRENT_SOURCE_DIR}/EnumStringConversions.h
    # ${CURRENT_SOURCE_DIR}/MayaCacheIO.cpp
    # ${CURRENT_SOURCE_DIR}/MayaCacheIO.h
    ${CURRENT_SOURCE_DIR}/SceneIO.cpp
    ${CURRENT_SOURCE_DIR}/SceneIO.h
    ${CURRENT_SOURCE_DIR}/SceneNodeI.h
    ${CURRENT_SOURCE_DIR}/SceneNodeNames.h)

set(SRC_INTERFACES
    ${CURRENT_SOURCE_DIR}/CommandCallbacksI.h
    ${CURRENT_SOURCE_DIR}/GPUSceneI.h
    ${CURRENT_SOURCE_DIR}/GPUTracerI.h
    ${CURRENT_SOURCE_DIR}/MovementSchemeI.h
    ${CURRENT_SOURCE_DIR}/NodeI.h
    ${CURRENT_SOURCE_DIR}/TracerCallbacksI.h
    ${CURRENT_SOURCE_DIR}/OptionsI.h
    ${CURRENT_SOURCE_DIR}/TracerSystemI.h
    ${CURRENT_SOURCE_DIR}/VisorCallbacksI.h
    ${CURRENT_SOURCE_DIR}/VisorI.h
    ${CURRENT_SOURCE_DIR}/VisorInputI.h)

set(SRC_MATH
    ${CURRENT_SOURCE_DIR}/AABB.cu
    ${CURRENT_SOURCE_DIR}/AABB.h
    ${CURRENT_SOURCE_DIR}/AABB.hpp
    ${CURRENT_SOURCE_DIR}/HemiDistribution.h
    ${CURRENT_SOURCE_DIR}/HybridFunctions.h
    ${CURRENT_SOURCE_DIR}/Matrix.cu
    ${CURRENT_SOURCE_DIR}/Matrix.h
    ${CURRENT_SOURCE_DIR}/Matrix.hpp
    ${CURRENT_SOURCE_DIR}/Quaternion.cu
    ${CURRENT_SOURCE_DIR}/Quaternion.h
    ${CURRENT_SOURCE_DIR}/Quaternion.hpp
    ${CURRENT_SOURCE_DIR}/Ray.cu
    ${CURRENT_SOURCE_DIR}/Ray.h
    ${CURRENT_SOURCE_DIR}/Ray.hpp
    ${CURRENT_SOURCE_DIR}/Vector.cu
    ${CURRENT_SOURCE_DIR}/Vector.h
    ${CURRENT_SOURCE_DIR}/Vector.hpp)

set(SRC_PRIMITIVES
    ${CURRENT_SOURCE_DIR}/Sphere.h
    ${CURRENT_SOURCE_DIR}/Triangle.h
    ${CURRENT_SOURCE_DIR}/Disk.h)


set(SRC_NODES
    ${CURRENT_SOURCE_DIR}/AnalyticNodeI.h
    ${CURRENT_SOURCE_DIR}/SelfNode.cpp
    ${CURRENT_SOURCE_DIR}/SelfNode.h
    ${CURRENT_SOURCE_DIR}/TracerNode.h
    ${CURRENT_SOURCE_DIR}/VisorNode.h)

set(SRC_STRUCTS
    ${CURRENT_SOURCE_DIR}/AnalyticData.h
    ${CURRENT_SOURCE_DIR}/ArrayPortion.h
    ${CURRENT_SOURCE_DIR}/HitStructs.h
    ${CURRENT_SOURCE_DIR}/ImageStructs.h
    ${CURRENT_SOURCE_DIR}/SceneStructs.h
    ${CURRENT_SOURCE_DIR}/TracerStatus.h
    ${CURRENT_SOURCE_DIR}/TracerStructs.h
    ${CURRENT_SOURCE_DIR}/VisorTransform.h
    ${CURRENT_SOURCE_DIR}/VisorInputStructs.h)

set(SRC_SURFACE_LOADER
    ${CURRENT_SOURCE_DIR}/BasicSurfaceLoaders.cpp
    ${CURRENT_SOURCE_DIR}/BasicSurfaceLoaders.h
    ${CURRENT_SOURCE_DIR}/SurfaceLoaderGenerator.cpp
    ${CURRENT_SOURCE_DIR}/SurfaceLoaderGenerator.h
    ${CURRENT_SOURCE_DIR}/SurfaceLoaderGeneratorI.h
    ${CURRENT_SOURCE_DIR}/SurfaceLoaderI.h
    ${CURRENT_SOURCE_DIR}/SurfaceLoaderPoolI.h)

set(SRC_SYSTEM
    ${CURRENT_SOURCE_DIR}/Endian.h
    ${CURRENT_SOURCE_DIR}/ObjectFuncDefinitions.h
    ${CURRENT_SOURCE_DIR}/SharedLib.cpp
    ${CURRENT_SOURCE_DIR}/SharedLib.h
    ${CURRENT_SOURCE_DIR}/System.h
    ${CURRENT_SOURCE_DIR}/Unreachable.h
    ${CURRENT_SOURCE_DIR}/Types.h)

set(SRC_THREADS
    ${CURRENT_SOURCE_DIR}/LoopingThreadI.h
    ${CURRENT_SOURCE_DIR}/ThreadVariable.h
    ${CURRENT_SOURCE_DIR}/TracerThread.cpp
    ${CURRENT_SOURCE_DIR}/TracerThread.h
    ${CURRENT_SOURCE_DIR}/VisorThread.cpp
    ${CURRENT_SOURCE_DIR}/VisorThread.h
    ${CURRENT_SOURCE_DIR}/WorkerThread.cpp
    ${CURRENT_SOURCE_DIR}/WorkerThread.h)

set(SRC_UTILITY
    ${CURRENT_SOURCE_DIR}/MortonCode.h
    ${CURRENT_SOURCE_DIR}/BitManipulation.cpp
    ${CURRENT_SOURCE_DIR}/BitManipulation.h
    ${CURRENT_SOURCE_DIR}/ColorConversion.h
    ${CURRENT_SOURCE_DIR}/CoordinateConversion.h
    ${CURRENT_SOURCE_DIR}/CPUTimer.h
    ${CURRENT_SOURCE_DIR}/FileSystemUtility.cpp
    ${CURRENT_SOURCE_DIR}/FileSystemUtility.h
    ${CURRENT_SOURCE_DIR}/MemoryAlignment.h
    ${CURRENT_SOURCE_DIR}/StripComments.h
    ${CURRENT_SOURCE_DIR}/TypeTraits.h
    ${CURRENT_SOURCE_DIR}/UTF8StringConversion.h
    ${CURRENT_SOURCE_DIR}/RandomColor.h
    ${CURRENT_SOURCE_DIR}/StringUtility.h
    ${CURRENT_SOURCE_DIR}/Flag.h)

set(SRC_VISOR_INPUT
    ${CURRENT_SOURCE_DIR}/MovementSchemes.cpp
    ${CURRENT_SOURCE_DIR}/MovementSchemes.h)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/Constants.h
    ${CURRENT_SOURCE_DIR}/CudaCheck.h
    ${CURRENT_SOURCE_DIR}/Log.h
    ${CURRENT_SOURCE_DIR}/PrimitiveDataTypes.h
    ${CURRENT_SOURCE_DIR}/Options.cpp
    ${CURRENT_SOURCE_DIR}/Options.h)

set(SRC_ALL
    ${SRC_CPU_CONTAINERS}
    ${SRC_DISTRIBUTOR}
    ${SRC_ERRORS}
    ${SRC_FILE_IO}
    ${SRC_INTERFACES}
    ${SRC_MATH}
    ${SRC_NODES}
    ${SRC_STRUCTS}
    ${SRC_SURFACE_LOADER}
    ${SRC_SYSTEM}
    ${SRC_THREADS}
    ${SRC_UTILITY}
    ${SRC_VISOR_INPUT}
    ${SRC_COMMON}
    ${SRC_PRIMITIVES})

# IDE Filters
source_group("CPU Containers" FILES ${SRC_CPU_CONTAINERS})
source_group("Distributor" FILES ${SRC_DISTRIBUTOR})
source_group("Errors" FILES ${SRC_ERRORS})
source_group("File I-O" FILES ${SRC_FILE_IO})
source_group("Interfaces" FILES ${SRC_INTERFACES})
source_group("Math" FILES ${SRC_MATH})
source_group("Math/Primitives" FILES ${SRC_PRIMITIVES})
source_group("Nodes" FILES ${SRC_NODES})
source_group("Structs" FILES ${SRC_STRUCTS})
source_group("SurfaceLoader" FILES ${SRC_SURFACE_LOADER})
source_group("System" FILES ${SRC_SYSTEM})
source_group("Threads" FILES ${SRC_THREADS})
source_group("Utility" FILES ${SRC_UTILITY})
source_group("VisorInput" FILES ${SRC_VISOR_INPUT})
source_group("" FILES ${SRC_COMMON})

# DLL File
add_library(RayLib STATIC ${SRC_ALL})

target_link_libraries(RayLib
                      fmt::fmt
                      mray::meta_compile_opts
                      mray::cuda_extra_compile_opts)

set_target_properties(RayLib PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Copy DLLS
if(MSVC)
    add_custom_command(TARGET RayLib POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       "${MRAY_CONFIG_LIB_DIRECTORY}/spdlog$<$<CONFIG:Debug>:d>.dll"
                       "${MRAY_CONFIG_LIB_DIRECTORY}/ZLIB$<$<CONFIG:Debug>:d>1.dll"
                       ${MRAY_CONFIG_BIN_DIRECTORY})
endif()