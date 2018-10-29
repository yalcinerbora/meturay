#include "TracerGenerators.h"
#include "TracerLogics.cuh"
#include "Materials.cuh"

#include "TracerLib/GPUAcceleratorLinear.cuh"



template TracerBaseLogicI* TypeGenWrappers::TracerLogicConstruct<TracerBaseLogicI, TracerBasic>(GPUBaseAcceleratorI&,
																								const AcceleratorBatchMappings&,
																								const MaterialBatchMappings&,
																								const TracerOptions&);
template GPUBaseAcceleratorI* TypeGenWrappers::DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorLinear>();

template void TypeGenWrappers::DefaultDestruct(TracerBaseLogicI*);
template void TypeGenWrappers::DefaultDestruct(GPUBaseAcceleratorI*);

using namespace TypeGenWrappers;

#include "TracerLib/GPUAcceleratorI.h"
#include "TracerLib/TracerLogicI.h"

BasicTracerLogicGenerator::BasicTracerLogicGenerator()
	: tracerGenerator(TracerLogicConstruct<TracerBaseLogicI, TracerBasic>,
					  DefaultDestruct<TracerBaseLogicI>)
	, baseAccelGenerator(DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorLinear>,
						 DefaultDestruct<GPUBaseAcceleratorI>)
	, tracerLogic(nullptr, DefaultDestruct<TracerBaseLogicI>)
	, baseAccelerator(nullptr, DefaultDestruct<GPUBaseAcceleratorI>)
{
	// Add Basic Mat and Batch
	// Material Types
	matGroupGenerators.emplace(ColorMaterial::TypeName,
							   GPUMatGroupGen(DefaultConstruct<GPUMaterialGroupI, ColorMaterial>,
											  DefaultDestruct<GPUMaterialGroupI>));

	// Material Batches
	matBatchGenerators.emplace(ColorMatSphrBatch::TypeName,
							   GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, ColorMatSphrBatch>,
											  DefaultDestruct<GPUMaterialBatchI>));
	matBatchGenerators.emplace(ColorMatTriBatch::TypeName,
							   GPUMatBatchGen(MaterialBatchConstruct<GPUMaterialBatchI, ColorMatTriBatch>,
											  DefaultDestruct<GPUMaterialBatchI>));
}

SceneError BasicTracerLogicGenerator::GetBaseLogic(TracerBaseLogicI*&)
{
	return SceneError::OK;
}