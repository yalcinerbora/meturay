#include "TracerGenerators.h"
#include "TracerLogics.cuh"

#include "BasicMaterials.cuh"
#include "GIMaterials.cuh"

#include "TracerLib/GPUAcceleratorLinear.cuh"

#include "TracerLib/GPUAcceleratorI.h"
#include "TracerLib/TracerLogicI.h"

template TracerBaseLogicI* TypeGenWrappers::TracerLogicConstruct<TracerBaseLogicI, TracerBasic>(GPUBaseAcceleratorI&,
																								const AcceleratorBatchMappings&,
																								const MaterialBatchMappings&,
																								const TracerParameters&,
																								uint32_t,
																								const Vector2i,
																								const Vector2i);
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
	matGroupGenerators.emplace(BasicMat::TypeName,
							   GPUMatGroupGen(DefaultConstruct<GPUMaterialGroupI, BasicMat>,
											  DefaultDestruct<GPUMaterialGroupI>));
	matGroupGenerators.emplace(GIAlbedoMat::TypeName,
							   GPUMatGroupGen(DefaultConstruct<GPUMaterialGroupI, GIAlbedoMat>,
											  DefaultDestruct<GPUMaterialGroupI>));
	matGroupGenerators.emplace(ConstantBoundaryMat::TypeName,
							   GPUMatGroupGen(DefaultConstruct<GPUMaterialGroupI, ConstantBoundaryMat>,
											  DefaultDestruct<GPUMaterialGroupI>));
	// Material Batches
	matBatchGenerators.emplace(GIAlbedoTriBatch::TypeName,
							   GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, GIAlbedoTriBatch>,
											  DefaultDestruct<GPUMaterialBatchI>));
	matBatchGenerators.emplace(GIAlbedoSphrBatch::TypeName,
							   GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, GIAlbedoSphrBatch>,
											  DefaultDestruct<GPUMaterialBatchI>));
	//
	matBatchGenerators.emplace(BasicMatTriBatch::TypeName,
							   GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BasicMatTriBatch>,
											  DefaultDestruct<GPUMaterialBatchI>));
	matBatchGenerators.emplace(BasicMatSphrBatch::TypeName,
							   GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, BasicMatSphrBatch>,
											  DefaultDestruct<GPUMaterialBatchI>));
	//
	matBatchGenerators.emplace(ConstantBoundaryMat::TypeName,
							   GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, ConstantBoundaryMatBatch>,
											  DefaultDestruct<GPUMaterialBatchI>));
}