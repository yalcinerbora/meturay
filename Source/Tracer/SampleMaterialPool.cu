#include "SampleMaterialPool.h"

#include "BasicMaterials.cuh"
#include "SampleMaterials.cuh"

using namespace TypeGenWrappers;

SampleMaterialPool::SampleMaterialPool()
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


extern "C" _declspec(dllexport) MaterialLogicPoolI * __stdcall GenerateSampleMaterialPool()
{
    return new SampleMaterialPool();
}

extern "C" _declspec(dllexport) void __stdcall DeleteSampleMaterialPool(MaterialLogicPoolI * pool)
{
    delete pool;
}