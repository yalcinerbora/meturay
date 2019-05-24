#include "TracerGenerators.h"
#include "TracerLogics.cuh"

#include "BasicMaterials.cuh"
#include "GIMaterials.cuh"

#include "TracerLib/GPUAcceleratorLinear.cuh"

#include "TracerLib/GPUAcceleratorI.h"
#include "TracerLib/TracerLogicI.h"

template TracerBaseLogicI* TypeGenWrappers::TracerLogicConstruct<TracerBaseLogicI, TracerBasic>(GPUBaseAcceleratorI& ba,
                                                                                                AcceleratorGroupList&& ag,
                                                                                                AcceleratorBatchMappings&& ab,
                                                                                                MaterialGroupList&& mg,
                                                                                                MaterialBatchMappings&& mb,
                                                                                                //
                                                                                                const TracerParameters&,
                                                                                                uint32_t,
                                                                                                const Vector2i,
                                                                                                const Vector2i,
                                                                                                const HitKey);
template GPUBaseAcceleratorI* TypeGenWrappers::DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorLinear>();

template void TypeGenWrappers::DefaultDestruct(TracerBaseLogicI*);
template void TypeGenWrappers::DefaultDestruct(GPUBaseAcceleratorI*);

using namespace TypeGenWrappers;

BasicTracerLogicGenerator::BasicTracerLogicGenerator()
    : TracerLogicGenerator(GPUTracerGen(TracerLogicConstruct<TracerBaseLogicI, TracerBasic>,
                                        DefaultDestruct<TracerBaseLogicI>),
                           GPUTracerPtr(nullptr, DefaultDestruct<TracerBaseLogicI>))
{
    // Add Basic Mat and Batch
    // Material Types
    matGroupGenerators.emplace(BasicMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, BasicMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(BarycentricMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, BarycentricMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(SphericalMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, SphericalMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));
    matGroupGenerators.emplace(GIAlbedoMat::TypeName(),
                               GPUMatGroupGen(MaterialGroupConstruct<GPUMaterialGroupI, GIAlbedoMat>,
                                              DefaultDestruct<GPUMaterialGroupI>));

    // Material Batches
    // Basic
    matBatchGenerators.emplace(BasicMatTriBatch::TypeName(),
                               GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BasicMatTriBatch>,
                                              DefaultDestruct<GPUMaterialBatchI>));
    matBatchGenerators.emplace(BasicMatSphrBatch::TypeName(),
                               GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BasicMatSphrBatch>,
                                              DefaultDestruct<GPUMaterialBatchI>));
    matBatchGenerators.emplace(BasicMatEmptyBatch::TypeName(),
                               GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BasicMatEmptyBatch>,
                                              DefaultDestruct<GPUMaterialBatchI>));
    // Barycentric
    matBatchGenerators.emplace(BarycentricMatTriBatch::TypeName(),
                               GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BarycentricMatTriBatch>,
                                              DefaultDestruct<GPUMaterialBatchI>));
    // Spherical
    matBatchGenerators.emplace(SphericalMatSphrBatch::TypeName(),
                               GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, SphericalMatSphrBatch>,
                                              DefaultDestruct<GPUMaterialBatchI>));
    // GI Albedo
    matBatchGenerators.emplace(GIAlbedoTriBatch::TypeName(),
                               GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, GIAlbedoTriBatch>,
                                              DefaultDestruct<GPUMaterialBatchI>));
    matBatchGenerators.emplace(GIAlbedoSphrBatch::TypeName(),
                               GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, GIAlbedoSphrBatch>,
                                              DefaultDestruct<GPUMaterialBatchI>));
}