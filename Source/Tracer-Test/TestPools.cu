#include "TestPools.h"

#include "DirectTracer.h"
#include "PathTracer.h"

#include "BasicMaterials.cuh"
#include "SampleMaterials.cuh"

#include "TracerLib/GPUAcceleratorLinear.cuh"
#include "TracerLib/GPUAcceleratorI.h"

using namespace TypeGenWrappers;

TestMaterialPool::TestMaterialPool()
{
    // Add Basic Mat and Batch
    // Material Types
    materialGroupGenerators.emplace(ConstantMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, ConstantMat>,
                                                   DefaultDestruct<GPUMaterialGroupI>));
    materialGroupGenerators.emplace(BarycentricMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, BarycentricMat>,
                                                   DefaultDestruct<GPUMaterialGroupI>));
    materialGroupGenerators.emplace(SphericalMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, SphericalMat>,
                                                   DefaultDestruct<GPUMaterialGroupI>));
    // Sample Materials
    materialGroupGenerators.emplace(EmissiveMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, EmissiveMat>,
                                                   DefaultDestruct<GPUMaterialGroupI>));
    materialGroupGenerators.emplace(LambertMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, LambertMat>,
                                                   DefaultDestruct<GPUMaterialGroupI>));
    materialGroupGenerators.emplace(ReflectMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, ReflectMat>,
                                                   DefaultDestruct<GPUMaterialGroupI>));
    materialGroupGenerators.emplace(RefractMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, RefractMat>,
                                                   DefaultDestruct<GPUMaterialGroupI>));
}

TestTracerPool::TestTracerPool()
{
    tracerGenerators.emplace(DirectTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, DirectTracer>,
                                          DefaultDestruct<GPUTracerI>));
    tracerGenerators.emplace(PathTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, PathTracer>,
                                          DefaultDestruct<GPUTracerI>));
}