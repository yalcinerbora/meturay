#include "TracerGenerators.h"
#include "TracerLogics.cuh"
#include "Materials.cuh"

#include "TracerLib/GPUAcceleratorLinear.cuh"

#include "TracerLib/GPUAcceleratorI.h"
#include "TracerLib/TracerLogicI.h"

template TracerBaseLogicI* TypeGenWrappers::TracerLogicConstruct<TracerBaseLogicI, TracerBasic>(GPUBaseAcceleratorI&,
																								const AcceleratorBatchMappings&,
																								const MaterialBatchMappings&,
																								const TracerOptions&);
template GPUBaseAcceleratorI* TypeGenWrappers::DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorLinear>();

template void TypeGenWrappers::DefaultDestruct(TracerBaseLogicI*);
template void TypeGenWrappers::DefaultDestruct(GPUBaseAcceleratorI*);

using namespace TypeGenWrappers;

BasicTracerLogicGenerator::BasicTracerLogicGenerator()
	: tracerGenerator(TracerLogicConstruct<TracerBaseLogicI, TracerBasic>,
					  DefaultDestruct<TracerBaseLogicI>)
	, tracerLogic(nullptr, DefaultDestruct<TracerBaseLogicI>)
{
	// Add Basic Mat and Batch
	// Material Types
	matGroupGenerators.emplace(ConstantAlbedoMat::TypeName,
							   GPUMatGroupGen(DefaultConstruct<GPUMaterialGroupI, ConstantAlbedoMat>,
											  DefaultDestruct<GPUMaterialGroupI>));
	matGroupGenerators.emplace(ConstantBoundaryMat::TypeName,
							   GPUMatGroupGen(DefaultConstruct<GPUMaterialGroupI, ConstantBoundaryMat>,
											  DefaultDestruct<GPUMaterialGroupI>));

	// Material Batches
	matBatchGenerators.emplace(ConstantAlbedoTriBatch::TypeName,
							   GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, ConstantAlbedoTriBatch>,
											  DefaultDestruct<GPUMaterialBatchI>));
	matBatchGenerators.emplace(ConstantAlbedoSphrBatch::TypeName,
							   GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, ConstantAlbedoSphrBatch>,
											  DefaultDestruct<GPUMaterialBatchI>));
	matBatchGenerators.emplace(ConstantBoundaryMat::TypeName,
							   GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, ConstantBoundaryMatBatch>,
											  DefaultDestruct<GPUMaterialBatchI>));

}

SceneError BasicTracerLogicGenerator::GetBaseLogic(TracerBaseLogicI*&)
{
	assert(false);
	return SceneError::OK;
}