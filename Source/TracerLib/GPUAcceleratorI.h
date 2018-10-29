#pragma once
/**

Base Interface for GPU accelerators

*/

#include <map>
#include <set>
#include <array>
#include <cstdint>
#include <functional>

#include "HitStructs.cuh"

struct RayGMem;
struct SceneError;
struct SceneFileNode;

class GPUPrimitiveGroupI;
class GPUMaterialGroupI;

// Accelerator Group defines same type of accelerators
// This struct holds accelerator data
// Unlike material group there is one to one relationship between
// Accelerator batch and group since accelerator is strongly tied with primitive
// However interface is split for consistency (to be like material group/batch)
class GPUAcceleratorGroupI
{
	public:
		virtual					~GPUAcceleratorGroupI() = default;

		// Interface
		// Type(as string) of the accelerator group
		virtual const char*		Type() const = 0;
		// Loads required data to CPU cache for
		virtual SceneError		InitializeGroup(SceneFileNode acceleratorNode,
												// Map of hit keys for all materials
												// can be queried by material file id
												const std::map<uint32_t, HitKey>&,
												// List of surface/material
												// pairings that uses this accelerator type
												// and primitive type
												const std::set<IdPairings>& pairingList,
												double time) = 0;
		virtual SceneError		ChangeTime(SceneFileNode acceleratorNode,
										   // Map of hit keys for all materials
										   // can be queried by material file id
										   const std::map<uint32_t, HitKey>&,
										   // List of surface/material
										   // pairings that uses this accelerator type
										   // and primitive type
										   const std::set<IdPairings>& pairingList,
										   double time) = 0;
		// Surface Queries
		virtual int							InnerId(uint32_t surface) = 0;

		// Batched and singular construction
		virtual void						ConstructAccelerator(uint32_t surface) = 0;
		virtual void						ConstructAccelerators(const std::vector<uint32_t>& surfaces) = 0;
		virtual void						DestroyAccelerator(uint32_t surface) = 0;
		virtual void						DestroyAccelerators(const std::vector<uint32_t>& surfaces) = 0;

		virtual size_t						UsedGPUMemory() const = 0;
		virtual size_t						UsedCPUMemory() const = 0;

		virtual const GPUPrimitiveGroupI&	PrimitiveGroup() const = 0;
};

//
class GPUAcceleratorBatchI
{
	public:
		virtual									~GPUAcceleratorBatchI() = default;

		// Interface
		// Type(as string) of the accelerator group
		virtual const char*						Type() const = 0;
		// Kernel Logic
		virtual void							Hit(// O
													HitKey* dMaterialKeys,
													PrimitiveId* dPrimitiveIds,
													HitStructPtr dHitStructs,
													// I-O													
													RayGMem* dRays,																									
													// Input
													const TransformId* dTransformIds,
													const RayId* dRayIds,
													const HitKey* dAcceleratorKeys,
													const uint32_t rayCount) const = 0;
		
		// Every MaterialBatch is available for a specific primitive / accelerator data
		virtual const GPUPrimitiveGroupI&		PrimitiveGroup() const = 0;
		virtual const GPUAcceleratorGroupI&		AcceleratorGroup() const = 0;
};

class GPUBaseAcceleratorI
{
	public:
		virtual					~GPUBaseAcceleratorI() = default;

		// Interface
		// Type(as string) of the accelerator group
		virtual const char*		Type() const = 0;
		// Base accelerator only points to the next accelerator key.
		// It can return invalid key,
		// which is either means data is out of bounds or ray is invalid.
		virtual void			Hit(// Output
									TransformId* dTransformIds,
									HitKey* dAcceleratorKeys,
									// Inputs
									const RayGMem* dRays,									
									const RayId* dRayIds,
									const uint32_t rayCount) const = 0;

		//TODO: define params of functions
		virtual void			Constrcut() = 0;
		virtual void			Reconstruct() = 0;
};