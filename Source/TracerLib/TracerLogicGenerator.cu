#include "TracerLogicGenerator.h"

#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveTriangle.h"
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
	: outerIdAccel(1)
	, outerIdMaterial(2)
	, baseAccelerator(nullptr, TypeGenWrappers::DefaultDestruct<GPUBaseAcceleratorI>)
{
	using namespace TypeGenWrappers;

	// Primitive Defaults
	primGroupGenerators.emplace(GPUPrimitiveTriangle::TypeName,
								GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveTriangle>,
												DefaultDestruct<GPUPrimitiveGroupI>));
	primGroupGenerators.emplace(GPUPrimitiveSphere::TypeName,
								GPUPrimGroupGen(DefaultConstruct<GPUPrimitiveGroupI, GPUPrimitiveSphere>,
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

SceneError TracerLogicGenerator::GetAcceleratorBatch(GPUAcceleratorBatchI*& ab, uint32_t& id,
													 const GPUAcceleratorGroupI& ag,
													 const GPUPrimitiveGroupI& pg)
{
	ab = nullptr;
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
		accelBatchMap.emplace(outerIdAccel, ab);
		id = outerIdAccel;
		outerIdAccel++;
	}
	else ab = loc->second.get();
	return SceneError::OK;
}

SceneError TracerLogicGenerator::GetMaterialBatch(GPUMaterialBatchI*& mb, uint32_t& id,
												  const GPUMaterialGroupI& mg,
												  const GPUPrimitiveGroupI& pg)
{
	mb = nullptr;
	const std::string batchType = std::string(mg.Type()) + pg.Type();

	auto loc = matBatches.find(batchType);
	if(loc == matBatches.end())
	{
		// Cannot Find Already Constructed Type
		// Generate
		auto loc = matBatchGenerators.find(batchType);
		if(loc == matBatchGenerators.end()) return SceneError::NO_LOGIC_FOR_MATERIAL;

		GPUMatBPtr ptr = loc->second(mg, pg);
		mb = ptr.get();
		matBatches.emplace(batchType, std::move(ptr));
		matBatchMap.emplace(outerIdMaterial, mb);
		id = outerIdMaterial;
		outerIdMaterial++;
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