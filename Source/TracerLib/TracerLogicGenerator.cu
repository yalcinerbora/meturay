#include "TracerLogicGenerator.h"

#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveEmpty.h"
#include "GPUAcceleratorLinear.cuh"

#include "GPUMaterialI.h"

// Default Generations
extern template class GPUAccLinearGroup<GPUPrimitiveTriangle>;
extern template class GPUAccLinearGroup<GPUPrimitiveSphere>;
extern template class GPUAcceleratorBatch<GPUAccLinearGroup<GPUPrimitiveTriangle>, GPUPrimitiveTriangle>;
extern template class GPUAcceleratorBatch<GPUAccLinearGroup<GPUPrimitiveSphere>, GPUPrimitiveSphere>;

// Typedefs for ease of read
using GPUAccTriLinearGroup = GPUAccLinearGroup<GPUPrimitiveTriangle>;
using GPUAccSphrLinearGroup = GPUAccLinearGroup<GPUPrimitiveSphere>;

using GPUAccTriLinearBatch = GPUAccLinearBatch<GPUPrimitiveTriangle>;
using GPUAccSphrLinearBatch = GPUAccLinearBatch<GPUPrimitiveSphere>;

// Some Instantiations
// Constructors
template GPUPrimitiveGroupI* TypeGenWrappers::DefaultConstruct<GPUPrimitiveGroupI,
															   GPUPrimitiveTriangle>();
template GPUPrimitiveGroupI* TypeGenWrappers::DefaultConstruct<GPUPrimitiveGroupI,
															   GPUPrimitiveSphere>();

template GPUAcceleratorGroupI* TypeGenWrappers::AccelGroupConstruct<GPUAcceleratorGroupI, 
																	GPUAccTriLinearGroup>(const GPUPrimitiveGroupI&,
																						  const TransformStruct*);
template GPUAcceleratorGroupI* TypeGenWrappers::AccelGroupConstruct<GPUAcceleratorGroupI, 
																	GPUAccSphrLinearGroup>(const GPUPrimitiveGroupI&,
																						   const TransformStruct*);

template GPUAcceleratorBatchI* TypeGenWrappers::AccelBatchConstruct<GPUAcceleratorBatchI, 
																	GPUAccTriLinearBatch>(const GPUAcceleratorGroupI&,
																						  const GPUPrimitiveGroupI&);
template GPUAcceleratorBatchI* TypeGenWrappers::AccelBatchConstruct<GPUAcceleratorBatchI, 
																	GPUAccSphrLinearBatch>(const GPUAcceleratorGroupI&,
																						   const GPUPrimitiveGroupI&);
// Destructors
template void TypeGenWrappers::DefaultDestruct(GPUPrimitiveGroupI*);
template void TypeGenWrappers::DefaultDestruct(GPUMaterialGroupI*);
template void TypeGenWrappers::DefaultDestruct(GPUAcceleratorGroupI*);

template void TypeGenWrappers::DefaultDestruct(GPUAcceleratorBatchI*);
template void TypeGenWrappers::DefaultDestruct(GPUMaterialBatchI*);

// Constructor & Destructor
TracerLogicGenerator::TracerLogicGenerator()
	: baseAccelerator(nullptr, TypeGenWrappers::DefaultDestruct<GPUBaseAcceleratorI>)
	, outsideMaterial(nullptr, TypeGenWrappers::DefaultDestruct<GPUMaterialGroupI>)
	, outsideMatBatch(nullptr, TypeGenWrappers::DefaultDestruct<GPUMaterialBatchI>)
	, emptyPrimitive(nullptr)
{
	using namespace TypeGenWrappers;

	// Primitive Defaults
	primGroupGenerators.emplace(GPUPrimitiveTriangle::TypeName,
								GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveTriangle>,
												DefaultDestruct<GPUPrimitiveGroupI>));
	primGroupGenerators.emplace(GPUPrimitiveSphere::TypeName,
								GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveSphere>,
												DefaultDestruct<GPUPrimitiveGroupI>));
	primGroupGenerators.emplace(GPUPrimitiveEmpty::TypeName,
								GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveEmpty>,
												DefaultDestruct<GPUPrimitiveGroupI>));

	// Accelerator Types
	accelGroupGenerators.emplace(GPUAccTriLinearGroup::TypeName,
								 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccTriLinearGroup>,
												  DefaultDestruct<GPUAcceleratorGroupI>));
	accelGroupGenerators.emplace(GPUAccSphrLinearGroup::TypeName,
								 GPUAccelGroupGen(AccelGroupConstruct<GPUAcceleratorGroupI, GPUAccSphrLinearGroup>,
												  DefaultDestruct<GPUAcceleratorGroupI>));

	accelBatchGenerators.emplace(GPUAccTriLinearBatch::TypeName,
								 GPUAccelBatchGen(AccelBatchConstruct<GPUAcceleratorBatchI, GPUAccTriLinearBatch>,
												  DefaultDestruct<GPUAcceleratorBatchI>));
	accelBatchGenerators.emplace(GPUAccSphrLinearBatch::TypeName,
								 GPUAccelBatchGen(AccelBatchConstruct<GPUAcceleratorBatchI, GPUAccSphrLinearBatch>,
												  DefaultDestruct<GPUAcceleratorBatchI>));

	// Base Accelerator
	baseAccelGenerators.emplace(GPUBaseAcceleratorLinear::TypeName,
								GPUBaseAccelGen(DefaultConstruct<GPUBaseAcceleratorI, GPUBaseAcceleratorLinear>,
												DefaultDestruct<GPUBaseAcceleratorI>));


	// Inistantiate empty primitive since it is used by outside material	
	const std::string emptyTypeName = GPUPrimitiveEmpty::TypeName;
	auto loc = primGroupGenerators.find(emptyTypeName);
	GPUPrimGPtr ptr = loc->second();
	emptyPrimitive = ptr.get();
	primGroups.emplace(emptyTypeName, std::move(ptr));

	// Default Types are loaded
	// Other Types are strongly tied to base tracer logic
	// i.e. Auxiliary Struct Etc.
}

// Groups
SceneError TracerLogicGenerator::GetPrimitiveGroup(GPUPrimitiveGroupI*& pg,
												   const std::string& primitiveType)
{
	pg = nullptr;
	auto loc = primGroups.find(primitiveType);
	if(loc == primGroups.end())
	{
		// Cannot Find Already Constructed Type
		// Generate
		auto loc = primGroupGenerators.find(primitiveType);
		if(loc == primGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_PRIMITIVE;

		GPUPrimGPtr ptr = loc->second();
		pg = ptr.get();
		primGroups.emplace(primitiveType, std::move(ptr));
	}
	else pg = loc->second.get();
	return SceneError::OK;
}

SceneError TracerLogicGenerator::GetAcceleratorGroup(GPUAcceleratorGroupI*& ag,
													 const GPUPrimitiveGroupI& pg,
													 const TransformStruct* t,
													 const std::string& accelType)
{
	ag = nullptr;
	auto loc = accelGroups.find(accelType);
	if(loc != accelGroups.end())
	{
		// Cannot Find Already Constructed Type
		// Generate
		auto loc = accelGroupGenerators.find(accelType);
		if(loc == accelGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_ACCELERATOR;

		GPUAccelGPtr ptr = loc->second(pg, t);
		ag = ptr.get();
		accelGroups.emplace(accelType, std::move(ptr));
	}
	else ag = loc->second.get();
	return SceneError::OK;
}

SceneError TracerLogicGenerator::GetMaterialGroup(GPUMaterialGroupI*& mg,												 
												  const std::string& materialType)
{
	mg = nullptr;
	auto loc = matGroups.find(materialType);
	if(loc == matGroups.end())
	{
		// Cannot Find Already Constructed Type
		// Generate
		auto loc = matGroupGenerators.find(materialType);
		if(loc == matGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_MATERIAL;

		GPUMatGPtr ptr = loc->second();
		mg = ptr.get();
		matGroups.emplace(materialType, std::move(ptr));
	}
	else mg = loc->second.get();
	return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateAcceleratorBatch(GPUAcceleratorBatchI*& ab,
														  const GPUAcceleratorGroupI& ag,
														  const GPUPrimitiveGroupI& pg,
														  uint32_t keyBatchId)
{
	ab = nullptr;
	// Check duplicate batchId
	if(accelBatchMap.find(keyBatchId) != accelBatchMap.end())
		return SceneError::INTERNAL_DUPLICATE_ACCEL_ID;
	
	const std::string batchType = std::string(ag.Type()) + pg.Type();

	auto loc = accelBatches.find(batchType);
	if(loc == accelBatches.end())
	{
		// Cannot Find Already Constructed Type
		// Generate
		auto loc = accelBatchGenerators.find(batchType);
		if(loc == accelBatchGenerators.end()) return SceneError::NO_LOGIC_FOR_ACCELERATOR;

		GPUAccelBPtr ptr = loc->second(ag, pg);
		ab = ptr.get();
		accelBatches.emplace(batchType, std::move(ptr));
		accelBatchMap.emplace(keyBatchId, ab);
	}
	else ab = loc->second.get();
	return SceneError::OK;
}

SceneError TracerLogicGenerator::GenerateMaterialBatch(GPUMaterialBatchI*& mb,
													   const GPUMaterialGroupI& mg,
													   const GPUPrimitiveGroupI& pg,
													   uint32_t keyBatchId,
													   int gpuId)
{
	mb = nullptr;
	if(matBatchMap.find(keyBatchId) != matBatchMap.end())
		return SceneError::INTERNAL_DUPLICATE_MAT_ID;
	
	const std::string batchType = std::string(mg.Type()) + pg.Type();

	auto loc = matBatches.find(std::make_pair(batchType, gpuId));
	if(loc == matBatches.end())
	{
		// Cannot Find Already Constructed Type
		// Generate
		auto loc = matBatchGenerators.find(batchType);
		if(loc == matBatchGenerators.end()) return SceneError::NO_LOGIC_FOR_MATERIAL;

		GPUMatBPtr ptr = loc->second(mg, pg, gpuId);
		mb = ptr.get();
		matBatches.emplace(std::make_pair(batchType, gpuId), std::move(ptr));
		matBatchMap.emplace(keyBatchId, mb);
	}
	else mb = loc->second.get();
	return SceneError::OK;
}

SceneError TracerLogicGenerator::GetBaseAccelerator(GPUBaseAcceleratorI*& baseAccel,
													const std::string& accelType)
{
	if(baseAccelerator.get() == nullptr)
	{
		// Cannot Find Already Constructed Type
		// Generate
		auto loc = baseAccelGenerators.find(accelType);
		if(loc == baseAccelGenerators.end()) return SceneError::NO_LOGIC_FOR_ACCELERATOR;		
		baseAccelerator = loc->second();		
		baseAccel = baseAccelerator.get();
	}
	else baseAccel = baseAccelerator.get();
	return SceneError::OK;
}

SceneError TracerLogicGenerator::GetOutsideMaterial(GPUMaterialGroupI*& outMat,
													const std::string& materialType, 
													int gpuId)
{
	if(outsideMaterial.get() == nullptr)
	{
		// Cannot Find Already Constructed Type
		// Generate
		auto loc = matGroupGenerators.find(materialType);
		if(loc == matGroupGenerators.end()) return SceneError::NO_LOGIC_FOR_MATERIAL;
		outsideMaterial = loc->second();
		outMat = outsideMaterial.get();

		// Additionally Generate a batch for it
		auto batchLoc = matBatchGenerators.find(materialType);
		if(batchLoc == matBatchGenerators.end()) return SceneError::NO_LOGIC_FOR_MATERIAL;

		outsideMatBatch = batchLoc->second(*outMat, *emptyPrimitive, gpuId);
		GPUMaterialBatchI* mb = outsideMatBatch.get();

		matBatchMap.emplace(BoundaryMatId, mb);
	}
	else outMat = outsideMaterial.get();
	return SceneError::OK;
}

SceneError TracerLogicGenerator::IncludeAcceleratorsFromDLL(const SharedLib&,
															const std::string& mangledName) const
{
	// TODO: Implement
	return SceneError::OK;
}

SceneError TracerLogicGenerator::IncludeMaterialsFromDLL(const SharedLib&,
														 const std::string& mangledName) const
{
	// TODO: Implement
	return SceneError::OK;
}

SceneError TracerLogicGenerator::IncludePrimitivesFromDLL(const SharedLib&,
														  const std::string& mangledName) const
{
	// TODO: Implement
	return SceneError::OK;
}