#pragma once

#include "RayLib/SceneStructs.h"

struct SceneError;
class SharedLib;

// Execution Related Abstraction
class GPUBaseAcceleratorI;
class GPUAcceleratorBatchI;
class GPUMaterialBatchI;
// Data Related Abstraction
class GPUPrimitiveGroupI;
class GPUAcceleratorGroupI;
class GPUMaterialGroupI;
// Base Logic
class TracerBaseLogicI;

class TracerLogicGeneratorI
{
	public:
		virtual					~TracerLogicGeneratorI() = default;

		// Logic Generators
		// This is the heart of the type generation mechanism
		// of the DLL (A.K.A abstract factory)
		// It generates or returns (if already constructed) types
		// w.r.t. a type name and parent type if applicable
		// Groups
		virtual SceneError		GetPrimitiveGroup(GPUPrimitiveGroupI*&,
												  const std::string& primitiveType) = 0;
		virtual SceneError		GetMaterialGroup(GPUMaterialGroupI*&,
												 const std::string& materialType) = 0;
		virtual SceneError		GetAcceleratorGroup(GPUAcceleratorGroupI*&,
													const GPUPrimitiveGroupI&,
													const TransformStruct* t,
													const std::string& accelType) = 0;
		
		// Batches are the abstraction of kernel calls
		// Each batch instance is equavilent to a kernel call	
		virtual SceneError		GetAcceleratorBatch(GPUAcceleratorBatchI*&, uint32_t& id,
													const GPUAcceleratorGroupI&,
													const GPUPrimitiveGroupI&) = 0;
		virtual SceneError		GetMaterialBatch(GPUMaterialBatchI*&, uint32_t& id,
												 const GPUMaterialGroupI&,
												 const GPUPrimitiveGroupI&) = 0;

		// Base Accelerator should be fetched after all the stuff is generated
		virtual SceneError		GetBaseAccelerator(GPUBaseAcceleratorI*&,
												   const std::string& accelType) = 0;

		// Finally get the tracer logic
		// Tracer logic will be constructed with respect to
		// Constructed batches
		virtual SceneError		GetBaseLogic(TracerBaseLogicI*&) = 0;
		
		// Inclusion Functionality
		// Additionally includes the materials from these libraries
		// No exclusion functionality provided just add what you need
		virtual SceneError		IncludeAcceleratorsFromDLL(const SharedLib&,
														   const std::string& mangledName = "\0") const = 0;
		virtual SceneError		IncludeMaterialsFromDLL(const SharedLib&,
														const std::string& mangledName = "\0") const = 0;
		virtual SceneError		IncludePrimitivesFromDLL(const SharedLib&,
														 const std::string& mangledName = "\0") const = 0;
};