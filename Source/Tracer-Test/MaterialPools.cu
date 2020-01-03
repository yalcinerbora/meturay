#include "MaterialPools.h"
#include "TracerLogics.cuh"

#include "BasicMaterials.cuh"
#include "GIMaterials.cuh"

#include "TracerLib/GPUAcceleratorLinear.cuh"

#include "TracerLib/GPUAcceleratorI.h"
#include "TracerLib/TracerLogicI.h"

//
//template TracerBaseLogicI* TypeGenWrappers::TracerLogicConstruct<TracerBaseLogicI, TracerBasic>(GPUBaseAcceleratorI& ba,
//                                                                                                AcceleratorGroupList&& ag,
//                                                                                                AcceleratorBatchMappings&& ab,
//                                                                                                MaterialGroupList&& mg,
//                                                                                                MaterialBatchMappings&& mb,
//                                                                                                //
//                                                                                                const TracerParameters&,
//                                                                                                uint32_t,
//                                                                                                const Vector2i,
//                                                                                                const Vector2i,
//                                                                                                const HitKey);
//
//template GPUBaseAcceleratorI* TypeGenWrappers::DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorLinear>();
//
//template void TypeGenWrappers::DefaultDestruct(TracerBaseLogicI*);
//template void TypeGenWrappers::DefaultDestruct(GPUBaseAcceleratorI*);

using namespace TypeGenWrappers;

TestMaterialPool::TestMaterialPool()
{
    // Add Basic Mat and Batch
    // Material Types
    materialGroupGenerators.emplace(BasicMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, BasicMat>,
                                                   DefaultDestruct<GPUMaterialGroupI>));
    materialGroupGenerators.emplace(BarycentricMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, BarycentricMat>,
                                                   DefaultDestruct<GPUMaterialGroupI>));
    materialGroupGenerators.emplace(SphericalMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, SphericalMat>,
                                                   DefaultDestruct<GPUMaterialGroupI>));
    materialGroupGenerators.emplace(BasicPathTraceMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, BasicPathTraceMat>,
                                                   DefaultDestruct<GPUMaterialGroupI>));
    materialGroupGenerators.emplace(LightBoundaryMat::TypeName(),
                                    GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, LightBoundaryMat>,
                                    DefaultDestruct<GPUMaterialGroupI>));

    // Material Batches
    // Basic
    materialBatchGenerators.emplace(BasicMatTriBatch::TypeName(),
                                    GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BasicMatTriBatch>,
                                                   DefaultDestruct<GPUMaterialBatchI>));
    materialBatchGenerators.emplace(BasicMatSphrBatch::TypeName(),
                                    GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BasicMatSphrBatch>,
                                                   DefaultDestruct<GPUMaterialBatchI>));
    materialBatchGenerators.emplace(BasicMatBatch::TypeName(),
                                    GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BasicMatBatch>,
                                                   DefaultDestruct<GPUMaterialBatchI>));
    // Barycentric
    materialBatchGenerators.emplace(BarycentricMatTriBatch::TypeName(),
                                    GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BarycentricMatTriBatch>,
                                                   DefaultDestruct<GPUMaterialBatchI>));
    // Spherical
    materialBatchGenerators.emplace(SphericalMatSphrBatch::TypeName(),
                                    GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, SphericalMatSphrBatch>,
                                                   DefaultDestruct<GPUMaterialBatchI>));
    // GI Albedo
    materialBatchGenerators.emplace(BasicPTSphereBatch::TypeName(),
                                    GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BasicPTSphereBatch>,
                                                   DefaultDestruct<GPUMaterialBatchI>));
    materialBatchGenerators.emplace(BasicPTTriangleBatch::TypeName(),
                                    GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BasicPTTriangleBatch>,
                                                   DefaultDestruct<GPUMaterialBatchI>));
    // Light Boundary
    materialBatchGenerators.emplace(LightBoundaryBatch::TypeName(),
                                    GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, LightBoundaryBatch>,
                                    DefaultDestruct<GPUMaterialBatchI>));
    materialBatchGenerators.emplace(LightBoundaryTriBatch::TypeName(),
                                    GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, LightBoundaryTriBatch>,
                                    DefaultDestruct<GPUMaterialBatchI>));
    materialBatchGenerators.emplace(LightBoundarySphrBatch::TypeName(),
                                    GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, LightBoundarySphrBatch>,
                                    DefaultDestruct<GPUMaterialBatchI>));
}