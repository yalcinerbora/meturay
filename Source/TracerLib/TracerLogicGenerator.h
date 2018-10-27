#pragma once
/**

Base implementation of Logic Generator

Provides storage of types (via across boundary capable unique ptrs)
and it adds default accelerators and primitives as default types.

*/

#include <map>

#include "TracerLogicGeneratorI.h"
#include "DefaultTypeGenerators.h"

class TracerLogicGenerator : public TracerLogicGeneratorI
{
	private:
	protected:
		// Type Generation Functions
		std::map<std::string, GPUPrimGroupGen>			primGroupGenerators;
		std::map<std::string, GPUAccelGroupGen>			accelGroupGenerators;		
		std::map<std::string, GPUMatGroupGen>			matGroupGenerators;

		std::map<std::string, GPUAccelBatchGen>			accelBatchGenerators;
		std::map<std::string, GPUMatBatchGen>			matBatchGenerators;

		// Generated Types
		// These hold ownership of classes (thus these will force destruction)
		std::map<std::string, GPUAccelGPtr>				accelGroups;
		std::map<std::string, GPUPrimGPtr>				primGroups;
		std::map<std::string, GPUMatGPtr>				matGroups;

		std::map<std::string, GPUAccelBPtr>				accelBatches;
		std::map<std::string, GPUMatBPtr>				matBatches;

		// Generated Batch Mappings
		std::map<uint32_t, GPUAcceleratorBatchI*>		accelBatchMap;
		std::map<uint32_t, GPUMaterialBatchI*>			matBatchMap;

	public:
		// Constructor & Destructor
									TracerLogicGenerator();
									TracerLogicGenerator(const TracerLogicGenerator&) = delete;
		TracerLogicGenerator&		operator=(const TracerLogicGenerator&) = delete;
									~TracerLogicGenerator() = default;

		// Groups
		SceneError					GetPrimitiveGroup(GPUPrimitiveGroupI*&,
													  const std::string& primitiveType) override;
		SceneError					GetMaterialGroup(GPUMaterialGroupI*&,
													 const std::string& materialType) override;
		SceneError					GetAcceleratorGroup(GPUAcceleratorGroupI*&,
														const GPUPrimitiveGroupI&,
														const TransformStruct* t,
														const std::string& accelType) override;

		// Batches are the abstraction of kernel calls
		// Each batch instance is equavilent to a kernel call	
		SceneError					GetAcceleratorBatch(GPUAcceleratorBatchI*&,
														const GPUAcceleratorGroupI&,
														const GPUPrimitiveGroupI&) override;
		SceneError					GetMaterialBatch(GPUMaterialBatchI*&,
													 const GPUMaterialGroupI&,
													 const GPUPrimitiveGroupI&) override;

		// Inclusion Functionality
		// Additionally includes the materials from these libraries
		// No exclusion functionality provided just add what you need
		SceneError					IncludeAcceleratorsFromDLL(const SharedLib&,
															   const std::string& mangledName = "\0") const override;
		SceneError					IncludeMaterialsFromDLL(const SharedLib&,
															const std::string& mangledName = "\0") const override;
		SceneError					IncludePrimitivesFromDLL(const SharedLib&,
															 const std::string& mangledName = "\0") const override;
};
