#include "TracerLogicGenerator.h"

#include "RayLib/DLLError.h"
// Primitives
#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveEmpty.h"
// Accelerators
#include "GPUAcceleratorLinear.cuh"
#include "GPUAcceleratorBVH.cuh"
#ifdef MRAY_OPTIX
    #include "GPUAcceleratorOptiX.cuh"
#endif
// Mediums
#include "GPUMediumHomogeneous.cuh"
#include "GPUMediumVacuum.cuh"
// Transforms
#include "GPUTransformSingle.cuh"
#include "GPUTransformIdentity.cuh"
// Materials
#include "DebugMaterials.cuh"
#include "SimpleMaterials.cuh"
#include "LambertMaterial.cuh"
#include "UnrealMaterial.cuh"
// Tracers
#include "DirectTracer.h"
#include "PathTracer.h"
#include "AOTracer.h"
#include "PPGTracer.h"
#include "RefPGTracer.h"
#include "RLTracer.h"
#include "WFPGTracer.h"
// Lights
#include "GPULightNull.cuh"
#include "GPULightConstant.cuh"
#include "GPULightPrimitive.cuh"
#include "GPULightDirectional.cuh"
#include "GPULightRectangular.cuh"
#include "GPULightPoint.cuh"
#include "GPULightSpot.cuh"
#include "GPULightDisk.cuh"
#include "GPULightSkySphere.cuh"
// Cameras
#include "GPUCameraPinhole.cuh"
#include "GPUCameraSpherical.cuh"
// Filters
#include "GPUReconFilterBox.h"
#include "GPUReconFilterTent.h"
#include "GPUReconFilterGaussian.h"
#include "GPUReconFilterMitchell.h"

// Type to utilize the generated ones
extern template class GPUAccLinearGroup<GPUPrimitiveTriangle>;
extern template class GPUAccLinearGroup<GPUPrimitiveSphere>;

extern template class GPUAccBVHGroup<GPUPrimitiveTriangle>;
extern template class GPUAccBVHGroup<GPUPrimitiveSphere>;

extern template class GPULight<GPUPrimitiveSphere>;
extern template class GPULight<GPUPrimitiveTriangle>;

extern template class CPULightGroup<GPUPrimitiveSphere>;
extern template class CPULightGroup<GPUPrimitiveTriangle>;

// Typedefs for ease of read
using GPUAccTriLinearGroup = GPUAccLinearGroup<GPUPrimitiveTriangle>;
using GPUAccSphrLinearGroup = GPUAccLinearGroup<GPUPrimitiveSphere>;

using GPUAccTriBVHGroup = GPUAccBVHGroup<GPUPrimitiveTriangle>;
using GPUAccSphrBVHGroup = GPUAccBVHGroup<GPUPrimitiveSphere>;
#ifdef MRAY_OPTIX
    extern template class GPUAccOptiXGroup<GPUPrimitiveTriangle>;
    extern template class GPUAccOptiXGroup<GPUPrimitiveSphere>;

    using GPUAccTriOptiXGroup = GPUAccOptiXGroup<GPUPrimitiveTriangle>;
    using GPUAccSphrOptiXGroup = GPUAccOptiXGroup<GPUPrimitiveSphere>;
#endif

// Some Instantiations
// Constructors
template GPUPrimitiveGroupI* TypeGenWrappers::DefaultConstruct<GPUPrimitiveGroupI,
                                                               GPUPrimitiveTriangle>();

template GPUAcceleratorGroupI* TypeGenWrappers::AccelGroupConstruct<GPUAcceleratorGroupI,
                                                                    GPUAccTriLinearGroup>(const GPUPrimitiveGroupI&);

// Destructor
template void TypeGenWrappers::DefaultDestruct(GPUPrimitiveGroupI*);
template void TypeGenWrappers::DefaultDestruct(GPUMaterialGroupI*);
template void TypeGenWrappers::DefaultDestruct(GPUAcceleratorGroupI*);

TracerLogicGenerator::TracerLogicGenerator()
{
    using namespace TypeGenWrappers;

    // Primitive Defaults
    primGroupGenerators.emplace(GPUPrimitiveTriangle::TypeName(),
                                GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveTriangle>,
                                                DefaultDestruct<GPUPrimitiveGroupI>));
    primGroupGenerators.emplace(GPUPrimitiveSphere::TypeName(),
                                GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveSphere>,
                                                DefaultDestruct<GPUPrimitiveGroupI>));
    primGroupGenerators.emplace(GPUPrimitiveEmpty::TypeName(),
                                GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveEmpty>,
                                                DefaultDestruct<GPUPrimitiveGroupI>));

    // Accelerator Types
    accelGroupGenerators.emplace(GPUAccTriLinearGroup::TypeName(),
                                 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccTriLinearGroup>,
                                                  DefaultDestruct<GPUAcceleratorGroupI>));
    accelGroupGenerators.emplace(GPUAccSphrLinearGroup::TypeName(),
                                 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccSphrLinearGroup>,
                                                  DefaultDestruct<GPUAcceleratorGroupI>));
    accelGroupGenerators.emplace(GPUAccTriBVHGroup::TypeName(),
                                 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccTriBVHGroup>,
                                                  DefaultDestruct<GPUAcceleratorGroupI>));
    accelGroupGenerators.emplace(GPUAccSphrBVHGroup::TypeName(),
                                 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccSphrBVHGroup>,
                                                  DefaultDestruct<GPUAcceleratorGroupI>));
    #ifdef MRAY_OPTIX
        accelGroupGenerators.emplace(GPUAccSphrOptiXGroup::TypeName(),
                                     GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccSphrOptiXGroup>,
                                                      DefaultDestruct<GPUAcceleratorGroupI>));
        accelGroupGenerators.emplace(GPUAccTriOptiXGroup::TypeName(),
                                     GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccTriOptiXGroup>,
                                                      DefaultDestruct<GPUAcceleratorGroupI>));
    #endif

    // Base Accelerator
    baseAccelGenerators.emplace(GPUBaseAcceleratorLinear::TypeName(),
                                GPUBaseAccelGen(DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorLinear>,
                                                DefaultDestruct<GPUBaseAcceleratorI>));
    baseAccelGenerators.emplace(GPUBaseAcceleratorBVH::TypeName(),
                                GPUBaseAccelGen(DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorBVH>,
                                                DefaultDestruct<GPUBaseAcceleratorI>));
    #ifdef MRAY_OPTIX
        baseAccelGenerators.emplace(GPUBaseAcceleratorOptiX::TypeName(),
                                    GPUBaseAccelGen(DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorOptiX>,
                                                    DefaultDestruct<GPUBaseAcceleratorI>));
    #endif
    // Material Types
    // Debug Materials
    matGroupGenerators.emplace(BarycentricMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, BarycentricMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(SphericalMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, SphericalMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(NormalRenderMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, NormalRenderMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(SphericalAnisoTestMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, SphericalAnisoTestMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    // Basic Materials
    matGroupGenerators.emplace(LambertCMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, LambertCMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(ReflectMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, ReflectMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(RefractMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, RefractMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    // Proper Materials
    matGroupGenerators.emplace(LambertMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, LambertMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(UnrealMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, UnrealMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));

    // Transform Types
    transGroupGenerators.emplace(CPUTransformIdentity::TypeName(),
                                 CPUTransformGen(DefaultConstruct<CPUTransformGroupI, CPUTransformIdentity>,
                                                 DefaultDestruct<CPUTransformGroupI>));
    transGroupGenerators.emplace(CPUTransformSingle::TypeName(),
                                 CPUTransformGen(DefaultConstruct<CPUTransformGroupI, CPUTransformSingle>,
                                                 DefaultDestruct<CPUTransformGroupI>));

    // Medium Types
    medGroupGenerators.emplace(CPUMediumVacuum::TypeName(),
                               CPUMediumGen(DefaultConstruct<CPUMediumGroupI, CPUMediumVacuum>,
                                            DefaultDestruct<CPUMediumGroupI>));
    medGroupGenerators.emplace(CPUMediumHomogeneous::TypeName(),
                               CPUMediumGen(DefaultConstruct<CPUMediumGroupI, CPUMediumHomogeneous>,
                                            DefaultDestruct<CPUMediumGroupI>));

    // Light Types
    lightGroupGenerators.emplace(CPULightGroupNull::TypeName(),
                                 CPULightGroupGen(LightGroupConstruct<CPULightGroupI, CPULightGroupNull>,
                                                  DefaultDestruct<CPULightGroupI>));
    lightGroupGenerators.emplace(CPULightGroupConstant::TypeName(),
                                 CPULightGroupGen(LightGroupConstruct<CPULightGroupI, CPULightGroupConstant>,
                                                  DefaultDestruct<CPULightGroupI>));
    lightGroupGenerators.emplace(CPULightGroupPoint::TypeName(),
                                 CPULightGroupGen(LightGroupConstruct<CPULightGroupI, CPULightGroupPoint>,
                                                  DefaultDestruct<CPULightGroupI>));
    lightGroupGenerators.emplace(CPULightGroupDirectional::TypeName(),
                                 CPULightGroupGen(LightGroupConstruct<CPULightGroupI, CPULightGroupDirectional>,
                                                  DefaultDestruct<CPULightGroupI>));
    lightGroupGenerators.emplace(CPULightGroupDisk::TypeName(),
                                 CPULightGroupGen(LightGroupConstruct<CPULightGroupI, CPULightGroupDisk>,
                                                  DefaultDestruct<CPULightGroupI>));
    lightGroupGenerators.emplace(CPULightGroupRectangular::TypeName(),
                                 CPULightGroupGen(LightGroupConstruct<CPULightGroupI, CPULightGroupRectangular>,
                                                  DefaultDestruct<CPULightGroupI>));
    lightGroupGenerators.emplace(CPULightGroupSpot::TypeName(),
                                 CPULightGroupGen(LightGroupConstruct<CPULightGroupI, CPULightGroupSpot>,
                                                  DefaultDestruct<CPULightGroupI>));
    lightGroupGenerators.emplace(CPULightGroupSkySphere::TypeName(),
                                 CPULightGroupGen(LightGroupConstruct<CPULightGroupI, CPULightGroupSkySphere>,
                                                  DefaultDestruct<CPULightGroupI>));
    lightGroupGenerators.emplace(CPULightGroup<GPUPrimitiveTriangle>::TypeName(),
                                 CPULightGroupGen(LightGroupConstruct<CPULightGroupI, CPULightGroup<GPUPrimitiveTriangle>>,
                                                  DefaultDestruct<CPULightGroupI>));
    lightGroupGenerators.emplace(CPULightGroup<GPUPrimitiveSphere>::TypeName(),
                                 CPULightGroupGen(LightGroupConstruct<CPULightGroupI, CPULightGroup<GPUPrimitiveSphere>>,
                                                  DefaultDestruct<CPULightGroupI>));

    // Camera Types
    camGroupGenerators.emplace(CPUCameraGroupPinhole::TypeName(),
                               CPUCameraGroupGen(CameraGroupConstruct<CPUCameraGroupI, CPUCameraGroupPinhole>,
                                                 DefaultDestruct<CPUCameraGroupI>));
    camGroupGenerators.emplace(CPUCameraGroupSpherical::TypeName(),
                               CPUCameraGroupGen(CameraGroupConstruct<CPUCameraGroupI, CPUCameraGroupSpherical>,
                                                 DefaultDestruct<CPUCameraGroupI>));
    // Tracers
    tracerGenerators.emplace(DirectTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, DirectTracer>,
                                          DefaultDestruct<GPUTracerI>));
    tracerGenerators.emplace(PathTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, PathTracer>,
                                          DefaultDestruct<GPUTracerI>));
    tracerGenerators.emplace(AOTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, AOTracer>,
                                          DefaultDestruct<GPUTracerI>));
    tracerGenerators.emplace(PPGTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, PPGTracer>,
                                          DefaultDestruct<GPUTracerI>));
    tracerGenerators.emplace(RefPGTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, RefPGTracer>,
                                          DefaultDestruct<GPUTracerI>));
    tracerGenerators.emplace(RLTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, RLTracer>,
                                          DefaultDestruct<GPUTracerI>));
    tracerGenerators.emplace(WFPGTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, WFPGTracer>,
                                          DefaultDestruct<GPUTracerI>));
    // Filters
    filterGenerators.emplace(GPUReconFilterBox::TypeName(),
                             GPUReconFilterGen(ReconFilterLogicConstruct<GPUReconFilterI, GPUReconFilterBox>,
                                               DefaultDestruct<GPUReconFilterI>));
    filterGenerators.emplace(GPUReconFilterTent::TypeName(),
                             GPUReconFilterGen(ReconFilterLogicConstruct<GPUReconFilterI, GPUReconFilterTent>,
                                               DefaultDestruct<GPUReconFilterI>));
    filterGenerators.emplace(GPUReconFilterGaussian::TypeName(),
                             GPUReconFilterGen(ReconFilterLogicConstruct<GPUReconFilterI, GPUReconFilterGaussian>,
                                               DefaultDestruct<GPUReconFilterI>));
    filterGenerators.emplace(GPUReconFilterMitchell::TypeName(),
                             GPUReconFilterGen(ReconFilterLogicConstruct<GPUReconFilterI, GPUReconFilterMitchell>,
                                               DefaultDestruct<GPUReconFilterI>));
}

SceneError TracerLogicGenerator::GeneratePrimitiveGroup(GPUPrimGPtr& pg,
                                                        const std::string& primitiveType)
{
    pg = nullptr;
    auto loc = primGroupGenerators.find(primitiveType);
    if(loc == primGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_PRIMITIVE;

    GPUPrimGPtr ptr = loc->second();
    pg = std::move(ptr);
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateAcceleratorGroup(GPUAccelGPtr& ag,
                                                          const GPUPrimitiveGroupI& pg,
                                                          const std::string& accelType)
{
    ag = nullptr;
    auto loc = accelGroupGenerators.find(accelType);
    if(loc == accelGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_ACCELERATOR;

    GPUAccelGPtr ptr = loc->second(pg);
    ag = std::move(ptr);
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateMaterialGroup(GPUMatGPtr& mg,
                                                       const CudaGPU& gpu,
                                                       const std::string& materialType)
{
    mg = nullptr;
    auto loc = matGroupGenerators.find(materialType);
    if(loc == matGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_MATERIAL;
    mg = std::move(loc->second(gpu));
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateBaseAccelerator(GPUBaseAccelPtr& baseAccel,
                                                         const std::string& accelType)
{
    auto loc = baseAccelGenerators.find(accelType);
    if(loc == baseAccelGenerators.end()) return SceneError::NO_LOGIC_FOR_ACCELERATOR;
    baseAccel = std::move(loc->second());
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateMediumGroup(CPUMediumGPtr& mg,
                                                     const std::string& mediumType)
{
    auto loc = medGroupGenerators.find(mediumType);
    if(loc == medGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_MEDIUM;
    mg = std::move(loc->second());
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateTransformGroup(CPUTransformGPtr& tg,
                                                        const std::string& transformType)
{
    auto loc = transGroupGenerators.find(transformType);
    if(loc == transGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_TRANSFORM;
    tg = std::move(loc->second());
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateCameraGroup(CPUCameraGPtr& cam,
                                                     const GPUPrimitiveGroupI* pg,
                                                     const std::string& cameraType)
{
    auto loc = camGroupGenerators.find(cameraType);
    if(loc == camGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_CAMERA;
    cam = std::move(loc->second(pg));
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateLightGroup(CPULightGPtr& light,
                                                    const CudaGPU& gpu,
                                                    const GPUPrimitiveGroupI* pg,
                                                    const std::string& lightType)
{
    auto loc = lightGroupGenerators.find(lightType);
    if(loc == lightGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_LIGHT;
    light = std::move(loc->second(pg, gpu));
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateTracer(GPUTracerPtr& tracerPtr,
                                                // Args
                                                const CudaSystem& s,
                                                const GPUSceneI& scene,
                                                const TracerParameters& p,
                                                // Type
                                                const std::string& tracerType)
{
    auto loc = tracerGenerators.find(tracerType);
    if(loc == tracerGenerators.end()) return SceneError::NO_LOGIC_FOR_TRACER;

    tracerPtr = std::move(loc->second(s, scene, p));
    return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateReconFilter(GPUReconFilterPtr& filterPtr,
                                                     float filterRadius,
                                                     const Options& filterOptions,
                                                     // Type
                                                     const std::string& filterType)
{
    auto loc = filterGenerators.find(filterType);
    if(loc == filterGenerators.end()) return SceneError::NO_LOGIC_FOR_RECON_FILTER;

    filterPtr = std::move(loc->second(filterRadius, filterOptions));
    return SceneError::OK;
}

//DLLError TracerLogicGenerator::IncludeBaseAcceleratorsFromDLL(const std::string& libName,
//                                                              const std::string& regex,
//                                                              const SharedLibArgs& mangledName)
//{
//    DLLError e = DLLError::OK;
//    SharedLib* lib = nullptr;
//    e = FindOrGenerateSharedLib(lib, libName);
//    if(e != DLLError::OK) return e;
//
//    BaseAcceleratorPoolPtr* pool = nullptr;
//    e = FindOrGeneratePool<BaseAcceleratorLogicPoolI>(pool, loadedBaseAccPools,
//                                                      {lib, mangledName});
//    if(e != DLLError::OK) return e;
//
//    auto map = (*pool)->BaseAcceleratorGenerators(regex);
//    baseAccelGenerators.insert(map.cbegin(), map.cend());
//    return e;
//    return DLLError::OK;
//    // Then
//}
//
//DLLError TracerLogicGenerator::IncludeAcceleratorsFromDLL(const std::string& libName,
//                                                          const std::string& regex,
//                                                          const SharedLibArgs& mangledName)
//{
//    DLLError e = DLLError::OK;
//    SharedLib* lib = nullptr;
//    e = FindOrGenerateSharedLib(lib, libName);
//    if(e != DLLError::OK) return e;
//
//    AcceleratorPoolPtr* pool = nullptr;
//    e = FindOrGeneratePool<AcceleratorLogicPoolI>(pool, loadedAccPools,
//                                                  {lib, mangledName});
//    if(e != DLLError::OK) return e;
//
//    auto groupMap = (*pool)->AcceleratorGroupGenerators(regex);
//    accelGroupGenerators.insert(groupMap.cbegin(), groupMap.cend());
//    return e;
//}
//
//DLLError TracerLogicGenerator::IncludeMaterialsFromDLL(const std::string& libName,
//                                                       const std::string& regex,
//                                                       const SharedLibArgs& mangledName)
//{
//    DLLError e = DLLError::OK;
//    SharedLib* lib = nullptr;
//    e = FindOrGenerateSharedLib(lib, libName);
//    if(e != DLLError::OK) return e;
//
//    MaterialPoolPtr* pool = nullptr;
//    e = FindOrGeneratePool<MaterialLogicPoolI>(pool, loadedMatPools,
//                                                {lib, mangledName});
//    if(e != DLLError::OK) return e;
//
//    auto groupMap = (*pool)->MaterialGroupGenerators(regex);
//    matGroupGenerators.insert(groupMap.cbegin(), groupMap.cend());
//    return e;
//}
//
//DLLError TracerLogicGenerator::IncludePrimitivesFromDLL(const std::string& libName,
//                                                        const std::string& regex,
//                                                        const SharedLibArgs& mangledName)
//{
//    DLLError e = DLLError::OK;
//    SharedLib* lib = nullptr;
//    e = FindOrGenerateSharedLib(lib, libName);
//    if(e != DLLError::OK) return e;
//
//    PrimitivePoolPtr* pool = nullptr;
//    e = FindOrGeneratePool<PrimitiveLogicPoolI>(pool, loadedPrimPools,
//                                                {lib, mangledName});
//    if(e != DLLError::OK) return e;
//
//    auto map = (*pool)->PrimitiveGenerators(regex);
//    primGroupGenerators.insert(map.cbegin(), map.cend());
//    return e;
//}
//
//DLLError TracerLogicGenerator::IncludeTracersFromDLL(const std::string& libName,
//                                                     const std::string& regex,
//                                                     const SharedLibArgs& mangledName)
//{
//    DLLError e = DLLError::OK;
//    SharedLib* lib = nullptr;
//    e = FindOrGenerateSharedLib(lib, libName);
//    if(e != DLLError::OK) return e;
//
//    TracerPoolPtr* pool = nullptr;
//    e = FindOrGeneratePool<TracerPoolI>(pool, loadedTracerPools,
//                                        {lib, mangledName});
//    if(e != DLLError::OK) return e;
//
//    auto map = (*pool)->TracerGenerators(regex);
//    tracerGenerators.insert(map.cbegin(), map.cend());
//    return e;
//}