
set(CURRENT_SOURCE_DIR ${MRAY_SOURCE_DIRECTORY}/TracerCUDA)

# SOURCES
set(SRC_ACCELS_BVH
    ${CURRENT_SOURCE_DIR}/GPUAcceleratorBVH.cu
    ${CURRENT_SOURCE_DIR}/GPUAcceleratorBVH.cuh
    ${CURRENT_SOURCE_DIR}/GPUAcceleratorBVH.hpp
    ${CURRENT_SOURCE_DIR}/GPUAcceleratorBVHKC.cuh)

set(SRC_ACCELS_LINEAR
    ${CURRENT_SOURCE_DIR}/GPUAcceleratorLinear.cu
    ${CURRENT_SOURCE_DIR}/GPUAcceleratorLinear.cuh
    ${CURRENT_SOURCE_DIR}/GPUAcceleratorLinear.hpp
    ${CURRENT_SOURCE_DIR}/GPUAcceleratorLinearKC.cuh)

set(SRC_ACCELS_COMMON
    ${CURRENT_SOURCE_DIR}/GPUAcceleratorCommonKC.cuh)

set(SRC_CAMERAS
    ${CURRENT_SOURCE_DIR}/GPUCameraPixel.cu
    ${CURRENT_SOURCE_DIR}/GPUCameraPixel.cuh
    ${CURRENT_SOURCE_DIR}/GPUCameraPinhole.cu
    ${CURRENT_SOURCE_DIR}/GPUCameraPinhole.cuh
    ${CURRENT_SOURCE_DIR}/GPUCameraSpherical.cu
    ${CURRENT_SOURCE_DIR}/GPUCameraSpherical.cuh)

set(SRC_CUDA_SYSTEM
    ${CURRENT_SOURCE_DIR}/CudaSystem.cpp
    ${CURRENT_SOURCE_DIR}/CudaSystem.h
    ${CURRENT_SOURCE_DIR}/CudaSystem.hpp
    ${CURRENT_SOURCE_DIR}/GPUEvent.h)

set(SRC_DEBUG
    ${CURRENT_SOURCE_DIR}/TracerDebug.cu
    ${CURRENT_SOURCE_DIR}/TracerDebug.h)

set(SRC_GPU_ALGORITHMS_COMMON
    ${CURRENT_SOURCE_DIR}/BinarySearch.cuh
    ${CURRENT_SOURCE_DIR}/ParallelPartition.cuh
    ${CURRENT_SOURCE_DIR}/ParallelReduction.cuh
    ${CURRENT_SOURCE_DIR}/ParallelScan.cuh
    ${CURRENT_SOURCE_DIR}/ParallelSequence.cuh
    ${CURRENT_SOURCE_DIR}/ParallelTransform.cuh
    ${CURRENT_SOURCE_DIR}/RadixSort.cuh
    ${CURRENT_SOURCE_DIR}/Random.cuh)

set(SRC_GPU_ALGORITHMS_FUNCTIONS
    ${CURRENT_SOURCE_DIR}/ComparisonFunctions.cuh
    ${CURRENT_SOURCE_DIR}/FunctionEnablers.cuh
    ${CURRENT_SOURCE_DIR}/ReduceFunctions.cuh
    ${CURRENT_SOURCE_DIR}/TransformFunctions.cuh)

set(SRC_GPU_ALGORITHMS_IMPLEMENTATIONS
    ${CURRENT_SOURCE_DIR}/ParallelReduction.cu
    ${CURRENT_SOURCE_DIR}/ParallelScanArithmetic.cu
    ${CURRENT_SOURCE_DIR}/ParallelScanMatrix2x2.cu
    ${CURRENT_SOURCE_DIR}/ParallelScanMatrix3x3.cu
    ${CURRENT_SOURCE_DIR}/ParallelScanMatrix4x4.cu
    ${CURRENT_SOURCE_DIR}/ParallelScanQuaternion.cu
    ${CURRENT_SOURCE_DIR}/ParallelScanVector.cu
    ${CURRENT_SOURCE_DIR}/RadixSort.cu)

set(SRC_GPU_CLASSES
    ${CURRENT_SOURCE_DIR}/GPUPiecewiseDistribution.cu
    ${CURRENT_SOURCE_DIR}/GPUPiecewiseDistribution.cuh
    ${CURRENT_SOURCE_DIR}/GPUSurface.h)

set(SRC_GPU_INTERFACES
    ${CURRENT_SOURCE_DIR}/GPUAcceleratorI.h
    ${CURRENT_SOURCE_DIR}/GPUCameraI.h
    ${CURRENT_SOURCE_DIR}/GPUDirectLightSamplerI.h
    ${CURRENT_SOURCE_DIR}/GPUEndpointI.h
    ${CURRENT_SOURCE_DIR}/GPULightI.h
    ${CURRENT_SOURCE_DIR}/GPUMaterialI.h
    ${CURRENT_SOURCE_DIR}/GPUMediumI.h
    ${CURRENT_SOURCE_DIR}/GPUPrimitiveI.h
    ${CURRENT_SOURCE_DIR}/GPUTransformI.h
    ${CURRENT_SOURCE_DIR}/GPUWorkI.h)

set(SRC_GPU_TEMPLATE_KERNELS
    ${CURRENT_SOURCE_DIR}/AuxiliaryDataKernels.cuh
    ${CURRENT_SOURCE_DIR}/GenerationKernels.cuh
    ${CURRENT_SOURCE_DIR}/WorkKernels.cuh)

set(SRC_GPU_TEMPLATE_FUNCTIONS
    ${CURRENT_SOURCE_DIR}/AcceleratorFunctions.h
    ${CURRENT_SOURCE_DIR}/CameraFunctions.h
    ${CURRENT_SOURCE_DIR}/MaterialFunctions.h
    ${CURRENT_SOURCE_DIR}/ImageFunctions.cuh)

set(SRC_LIGHTS
    ${CURRENT_SOURCE_DIR}/GPULightDirectional.cu
    ${CURRENT_SOURCE_DIR}/GPULightDirectional.cuh
    ${CURRENT_SOURCE_DIR}/GPULightDisk.cu
    ${CURRENT_SOURCE_DIR}/GPULightDisk.cuh
    ${CURRENT_SOURCE_DIR}/GPULightPoint.cu
    ${CURRENT_SOURCE_DIR}/GPULightPoint.cuh
    ${CURRENT_SOURCE_DIR}/GPULightPrimitive.cu
    ${CURRENT_SOURCE_DIR}/GPULightPrimitive.cuh
    ${CURRENT_SOURCE_DIR}/GPULightPrimitive.hpp
    ${CURRENT_SOURCE_DIR}/GPULightRectangular.cu
    ${CURRENT_SOURCE_DIR}/GPULightRectangular.cuh
    ${CURRENT_SOURCE_DIR}/GPULightSkySphere.cu
    ${CURRENT_SOURCE_DIR}/GPULightSkySphere.cuh
    ${CURRENT_SOURCE_DIR}/GPULightSpot.cu
    ${CURRENT_SOURCE_DIR}/GPULightSpot.cuh)

set(SRC_LIGHT_SAMPLERS
    ${CURRENT_SOURCE_DIR}/GPULightSamplerUniform.cuh)

set(SRC_MATERIALS
    ${CURRENT_SOURCE_DIR}/BoundaryMaterials.cu
    ${CURRENT_SOURCE_DIR}/BoundaryMaterials.cuh
    ${CURRENT_SOURCE_DIR}/BoundaryMaterialsKC.cuh
    ${CURRENT_SOURCE_DIR}/DebugMaterials.cu
    ${CURRENT_SOURCE_DIR}/DebugMaterials.cuh
    ${CURRENT_SOURCE_DIR}/DebugMaterialsKC.cuh
    ${CURRENT_SOURCE_DIR}/EmptyMaterial.cuh
    ${CURRENT_SOURCE_DIR}/EmptyMaterialKC.cuh
    ${CURRENT_SOURCE_DIR}/LambertMaterial.cu
    ${CURRENT_SOURCE_DIR}/LambertMaterial.cuh
    ${CURRENT_SOURCE_DIR}/LambertMaterialKC.cuh
    ${CURRENT_SOURCE_DIR}/UnrealMaterial.cu
    ${CURRENT_SOURCE_DIR}/UnrealMaterial.cuh
    ${CURRENT_SOURCE_DIR}/UnrealMaterialKC.cuh
    ${CURRENT_SOURCE_DIR}/SimpleMaterials.cu
    ${CURRENT_SOURCE_DIR}/SimpleMaterials.cuh
    ${CURRENT_SOURCE_DIR}/SimpleMaterialsKC.cuh
    ${CURRENT_SOURCE_DIR}/MaterialDataStructs.h
    ${CURRENT_SOURCE_DIR}/MetaMaterialFunctions.cuh)

set(SRC_MEDIUMS
    ${CURRENT_SOURCE_DIR}/GPUMediumHomogenous.cu
    ${CURRENT_SOURCE_DIR}/GPUMediumHomogenous.cuh
    ${CURRENT_SOURCE_DIR}/GPUMediumVacuum.cuh)

set(SRC_MEMORY
    ${CURRENT_SOURCE_DIR}/DeviceMemory.cpp
    ${CURRENT_SOURCE_DIR}/DeviceMemory.h
    ${CURRENT_SOURCE_DIR}/GPUBitmap.h
    ${CURRENT_SOURCE_DIR}/ImageMemory.cu
    ${CURRENT_SOURCE_DIR}/ImageMemory.h
    ${CURRENT_SOURCE_DIR}/RayMemory.cu
    ${CURRENT_SOURCE_DIR}/RayMemory.h
    ${CURRENT_SOURCE_DIR}/RNGMemory.cu
    ${CURRENT_SOURCE_DIR}/RNGMemory.h
    ${CURRENT_SOURCE_DIR}/Texture.cpp
    ${CURRENT_SOURCE_DIR}/Texture.cuh
    ${CURRENT_SOURCE_DIR}/Texture.hpp)

set(SRC_PARTIAL_IMPLEMENTATIONS
    ${CURRENT_SOURCE_DIR}/GPUAcceleratorP.cuh
    ${CURRENT_SOURCE_DIR}/GPUMaterialP.cuh
    ${CURRENT_SOURCE_DIR}/GPUPrimitiveP.cuh)

set(SRC_PRIMITIVES
    ${CURRENT_SOURCE_DIR}/DefaultLeaf.h
    ${CURRENT_SOURCE_DIR}/GPUPrimitiveEmpty.cpp
    ${CURRENT_SOURCE_DIR}/GPUPrimitiveEmpty.h
    ${CURRENT_SOURCE_DIR}/GPUPrimitiveTriangle.cpp
    ${CURRENT_SOURCE_DIR}/GPUPrimitiveTriangle.h
    ${CURRENT_SOURCE_DIR}/GPUPrimitiveSphere.cpp
    ${CURRENT_SOURCE_DIR}/GPUPrimitiveSphere.h)

set(SRC_SCENE
    ${CURRENT_SOURCE_DIR}/GPUSceneJson.cpp
    ${CURRENT_SOURCE_DIR}/GPUSceneJson.h
    ${CURRENT_SOURCE_DIR}/NodeListing.h
    ${CURRENT_SOURCE_DIR}/SceneNodeJson.cpp
    ${CURRENT_SOURCE_DIR}/SceneNodeJson.h
    ${CURRENT_SOURCE_DIR}/ScenePartitioner.cpp
    ${CURRENT_SOURCE_DIR}/ScenePartitioner.h
    ${CURRENT_SOURCE_DIR}/ScenePartitionerI.h)
set(SRC_SHARED_LIB
    ${CURRENT_SOURCE_DIR}/EntryPoint.cpp
    ${CURRENT_SOURCE_DIR}/EntryPoint.h
    ${CURRENT_SOURCE_DIR}/TracerLogicGenerator.cu
    ${CURRENT_SOURCE_DIR}/TracerLogicGenerator.h
    ${CURRENT_SOURCE_DIR}/TracerLogicGeneratorI.h
    ${CURRENT_SOURCE_DIR}/TracerSystemCUDA.cpp
    ${CURRENT_SOURCE_DIR}/TracerSystemCUDA.h
    ${CURRENT_SOURCE_DIR}/TracerTypeGenerators.h)

set(SRC_STRUCTS
    ${CURRENT_SOURCE_DIR}/ImageStructs.h
    ${CURRENT_SOURCE_DIR}/RayStructs.h
    ${CURRENT_SOURCE_DIR}/RNGStructs.h)

set(SRC_TEXTURES
    ${CURRENT_SOURCE_DIR}/TextureFunctions.cpp
    ${CURRENT_SOURCE_DIR}/TextureFunctions.h
    ${CURRENT_SOURCE_DIR}/TextureReference.cu
    ${CURRENT_SOURCE_DIR}/TextureReference.cuh
    ${CURRENT_SOURCE_DIR}/TextureReference.hpp
    ${CURRENT_SOURCE_DIR}/TextureReferenceGenerators.cuh)

set(SRC_TRACERS_COMMON
    ${CURRENT_SOURCE_DIR}/GPUTracer.cu
    ${CURRENT_SOURCE_DIR}/GPUTracer.h
    ${CURRENT_SOURCE_DIR}/RayTracer.cu
    ${CURRENT_SOURCE_DIR}/RayTracer.h
    ${CURRENT_SOURCE_DIR}/RayTracer.hpp
    ${CURRENT_SOURCE_DIR}/TracerFunctions.cuh
    ${CURRENT_SOURCE_DIR}/Tracers.cu
    ${CURRENT_SOURCE_DIR}/Tracers.cpp
    ${CURRENT_SOURCE_DIR}/Tracers.h)

set(SRC_TRACERS_STRUCTS
    ${CURRENT_SOURCE_DIR}/PathNode.cuh
    ${CURRENT_SOURCE_DIR}/RayAuxStruct.cuh)

set(SRC_AO_TRACER
    ${CURRENT_SOURCE_DIR}/AOTracer.cu
    ${CURRENT_SOURCE_DIR}/AOTracer.h
    ${CURRENT_SOURCE_DIR}/AOTracerKC.cuh
    ${CURRENT_SOURCE_DIR}/AOTracerWork.cu
    ${CURRENT_SOURCE_DIR}/AOTracerWork.cuh)

set(SRC_DIRECT_TRACER
    ${CURRENT_SOURCE_DIR}/DirectTracer.cu
    ${CURRENT_SOURCE_DIR}/DirectTracer.h
    ${CURRENT_SOURCE_DIR}/DirectTracerKC.cuh
    ${CURRENT_SOURCE_DIR}/DirectTracerWorks.cu
    ${CURRENT_SOURCE_DIR}/DirectTracerWorks.cuh)

set(SRC_PATH_TRACER
    ${CURRENT_SOURCE_DIR}/PathTracer.cu
    ${CURRENT_SOURCE_DIR}/PathTracer.h
    ${CURRENT_SOURCE_DIR}/PathTracerKC.cuh
    ${CURRENT_SOURCE_DIR}/PathTracerWorks.cu
    ${CURRENT_SOURCE_DIR}/PathTracerWorks.cuh)

set(SRC_PPG_TRACER
    ${CURRENT_SOURCE_DIR}/PPGTracer.cu
    ${CURRENT_SOURCE_DIR}/PPGTracer.h
    ${CURRENT_SOURCE_DIR}/PPGTracerKC.cuh
    ${CURRENT_SOURCE_DIR}/PPGTracerWork.cu
    ${CURRENT_SOURCE_DIR}/PPGTracerWork.cuh)

set(SRC_PPG_TRACER_AUX_CLASSES
    ${CURRENT_SOURCE_DIR}/DTree.cu
    ${CURRENT_SOURCE_DIR}/DTree.cuh
    ${CURRENT_SOURCE_DIR}/DTreeKC.cuh
    ${CURRENT_SOURCE_DIR}/STree.cu
    ${CURRENT_SOURCE_DIR}/STree.cuh
    ${CURRENT_SOURCE_DIR}/STreeKC.cuh)

set(SRC_REF_PG_TRACER
    ${CURRENT_SOURCE_DIR}/RefPGTracer.cu
    ${CURRENT_SOURCE_DIR}/RefPGTracer.h
    ${CURRENT_SOURCE_DIR}/RefPGTracerKC.cuh
    ${CURRENT_SOURCE_DIR}/RefPGTracerWorks.cu
    ${CURRENT_SOURCE_DIR}/RefPGTracerWorks.cuh)

set(SRC_TRANSFORMS
    ${CURRENT_SOURCE_DIR}/GPUTransformIdentity.cuh
    ${CURRENT_SOURCE_DIR}/GPUTransformSingle.cu
    ${CURRENT_SOURCE_DIR}/GPUTransformSingle.cuh)

set(SRC_WORK
    ${CURRENT_SOURCE_DIR}/GPUWork.cuh
    ${CURRENT_SOURCE_DIR}/WorkPool.h)

set(SRC_COMMON
    ${CURRENT_SOURCE_DIR}/MangledNames.cpp
    ${CURRENT_SOURCE_DIR}/MangledNames.h
    ${CURRENT_SOURCE_DIR}/TracerConstants.h
    ${CURRENT_SOURCE_DIR}/TypeTraits.h)

set(SRC_ALL
    ${SRC_ACCELS_BVH}
    ${SRC_ACCELS_LINEAR}
    ${SRC_ACCELS_COMMON}
    ${SRC_CAMERAS}
    ${SRC_CUDA_SYSTEM}
    ${SRC_DEBUG}
    ${SRC_GPU_ALGORITHMS_COMMON}
    ${SRC_GPU_ALGORITHMS_FUNCTIONS}
    ${SRC_GPU_ALGORITHMS_IMPLEMENTATIONS}
    ${SRC_GPU_CLASSES}
    ${SRC_GPU_INTERFACES}
    ${SRC_GPU_TEMPLATE_KERNELS}
    ${SRC_GPU_TEMPLATE_FUNCTIONS}
    ${SRC_LIGHTS}
    ${SRC_LIGHT_SAMPLERS}
    ${SRC_MATERIALS}
    ${SRC_MEDIUMS}
    ${SRC_MEMORY}
    ${SRC_PARTIAL_IMPLEMENTATIONS}
    ${SRC_PRIMITIVES}
    ${SRC_SCENE}
    ${SRC_SHARED_LIB}
    ${SRC_STRUCTS}
    ${SRC_TEXTURES}
    ${SRC_TRACERS_COMMON}
    ${SRC_TRACERS_STRUCTS}
    ${SRC_AO_TRACER}
    ${SRC_DIRECT_TRACER}
    ${SRC_PATH_TRACER}
    ${SRC_PPG_TRACER}
    ${SRC_PPG_TRACER_AUX_CLASSES}
    ${SRC_REF_PG_TRACER}
    ${SRC_TRANSFORMS}
    ${SRC_WORK}
    ${SRC_COMMON})

# IDE Filters
source_group("Accelerators" FILES ${SRC_ACCELS_COMMON})
source_group("Accelerators/BVH" FILES ${SRC_ACCELS_BVH})
source_group("Accelerators/Linear" FILES ${SRC_ACCELS_LINEAR})

source_group("Cameras" FILES ${SRC_CAMERAS})
source_group("CudaSystem" FILES ${SRC_CUDA_SYSTEM})
source_group("Debug" FILES ${SRC_DEBUG})

source_group("GPU Algorithms" FILES ${SRC_GPU_ALGORITHMS_COMMON})
source_group("GPU Algorithms/Functions" FILES ${SRC_GPU_ALGORITHMS_FUNCTIONS})
source_group("GPU Algorithms/Implementations" FILES ${SRC_GPU_ALGORITHMS_IMPLEMENTATIONS})

source_group("GPU Classes" FILES ${SRC_GPU_CLASSES})
source_group("GPU Interfaces" FILES ${SRC_GPU_INTERFACES})
source_group("GPU Template Kernels" FILES ${SRC_GPU_TEMPLATE_KERNELS})
source_group("GPU Template Functions" FILES ${SRC_GPU_TEMPLATE_FUNCTIONS})
source_group("Lights" FILES ${SRC_LIGHTS})
source_group("Light Samplers" FILES ${SRC_LIGHT_SAMPLERS})
source_group("Materials" FILES ${SRC_MATERIALS})
source_group("Mediums" FILES ${SRC_MEDIUMS})
source_group("Memory" FILES ${SRC_MEMORY})
source_group("Partial Implementations" FILES ${SRC_PARTIAL_IMPLEMENTATIONS})
source_group("Primitives" FILES ${SRC_PRIMITIVES})
source_group("Scene" FILES ${SRC_SCENE})
source_group("Shared Lib" FILES ${SRC_SHARED_LIB})
source_group("Structs" FILES ${SRC_STRUCTS})
source_group("Textures" FILES ${SRC_TEXTURES})

source_group("Tracers" FILES ${SRC_TRACERS_COMMON})
source_group("Tracers/Structs" FILES ${SRC_TRACERS_STRUCTS})
source_group("Tracers/AOTracer" FILES ${SRC_AO_TRACER})
source_group("Tracers/DirectTracer" FILES ${SRC_DIRECT_TRACER})
source_group("Tracers/PathTracer" FILES ${SRC_PATH_TRACER})
source_group("Tracers/PPGTracer" FILES ${SRC_PPG_TRACER})
source_group("Tracers/PPGTracer/Aux Classes" FILES ${SRC_PPG_TRACER_AUX_CLASSES})
source_group("Tracers/RefPGTracer" FILES ${SRC_REF_PG_TRACER})

source_group("Transforms" FILES ${SRC_TRANSFORMS})
source_group("Work" FILES ${SRC_WORK})

source_group("" FILES ${SRC_COMMON})

# Preprocessor Defs
set(PREPROCESSOR_GENERIC
    ${MRAY_PREPROCESSOR_DEFS_CUDA_GENERIC}
    METU_SHARED_TRACER_CUDA)

# TBB for std::exectuion (clang & GCC)
if(MSVC)
    set(PLATFORM_SPEC_LIBRARIES)
elseif(UNIX)
    set(PLATFORM_SPEC_LIBRARIES tbb)
endif()

# DLL File
add_library(TracerCUDA SHARED)

target_sources(TracerCUDA PUBLIC ${SRC_ALL})

target_include_directories(TracerCUDA PRIVATE
                           ${MRAY_LIB_INCLUDE_DIRECTORY}
                           ${MRAY_SOURCE_DIRECTORY})

target_compile_definitions(TracerCUDA PRIVATE ${PREPROCESSOR_GENERIC})

target_compile_options(TracerCUDA PRIVATE ${MRAY_COMPILER_FLAGS_GENERIC})

set_target_properties(TracerCUDA PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(TracerCUDA
                      RayLib
                      ImageIO
                      ${PLATFORM_SPEC_LIBRARIES})

# Object Group for testing
#add_library(TracerCUDA_ForTest STATIC ${SRC_ALL})
add_library(TracerCUDA_ForTest STATIC $<TARGET_OBJECTS:TracerCUDA>)

target_include_directories(TracerCUDA_ForTest PRIVATE
                           ${MRAY_LIB_INCLUDE_DIRECTORY}
                           ${MRAY_SOURCE_DIRECTORY})

target_compile_definitions(TracerCUDA_ForTest PRIVATE ${PREPROCESSOR_GENERIC})

target_compile_options(TracerCUDA_ForTest PRIVATE ${MRAY_COMPILER_FLAGS_GENERIC})

set_target_properties(TracerCUDA_ForTest PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_SEPARABLE_COMPILATION ON
                      CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(TracerCUDA_ForTest
                      RayLib
                      ImageIO
                      ${PLATFORM_SPEC_LIBRARIES})