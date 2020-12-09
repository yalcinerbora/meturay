#include "TracerLogicGeneratorAll.h"

#include "TracerCUDALib/TracerTypeGenerators.h"

// Materials
#include "BasicMaterials.cuh"
#include "SampleMaterials.cuh"
// Tracers
#include "PathTracer.h"
#include "DirectTracer.h"
// Accelerators
// Primitives
// Transforms
// Mediums
// Lights
// Cameras

TracerLogicGeneratorAll::TracerLogicGeneratorAll()
{
    using namespace TypeGenWrappers;

    // Add Basic Mat and Batch
    // Material Types
    matGroupGenerators.emplace(ConstantMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, ConstantMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(BarycentricMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, BarycentricMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(SphericalMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, SphericalMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    // Sample Materials
    matGroupGenerators.emplace(EmissiveMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, EmissiveMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(LambertMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, LambertMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(ReflectMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, ReflectMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(RefractMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, RefractMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));

    // Tracers
    tracerGenerators.emplace(DirectTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, DirectTracer>,
                                          DefaultDestruct<GPUTracerI>));
    tracerGenerators.emplace(PathTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, PathTracer>,
                                          DefaultDestruct<GPUTracerI>));
}