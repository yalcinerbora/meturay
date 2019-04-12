#pragma once
/**

Linear Accelerator Implementation

This is actually not an accelerator
it traverses the "constructed" (portionized)
group of primitives and calls intersection functions
one by one

It is here for sinple scenes and objects in which
tree constructio would provide additional overhead.

*/

#include <array>

#include "RayLib/SceneStructs.h"
#include "RayLib/Vector.h"
#include "RayLib/Constants.h"

#include "DeviceMemory.h"
#include "GPUAcceleratorP.cuh"
#include "GPUPrimitiveI.h"

#include "GPUAcceleratorLinearKC.cuh"

// This should be an array?
// Most of the time each accelerator will be constructred with a 
// Singular primitive batch, it should be better to put size constraint
//using SurfaceDataList = std::vector<uint32_t>;
using SurfaceMaterialPairs = std::array<Vector2ul, SceneConstants::MaxSurfacePerAccelerator>;

template<class PGroup>
class LinearAccelTypeName
{
	public:
		static const std::string TypeName;
};

template<class PGroup>
class GPUAccLinearBatch;

template <class PGroup>
class GPUAccLinearGroup final 
	: public GPUAcceleratorGroup<PGroup>
	, public LinearAccelTypeName<PGroup>
{
	public:
		using LeafData								= PGroup::LeafData;

	private:	
		// CPU Memory
		//std::vector<Vector2ul>					acceleratorRanges;
		std::vector<PrimitiveRangeList>				primitiveRanges;
		std::vector<HitKeyList>						primitiveMaterialKeys;
		std::map<uint32_t, uint32_t>				idLookup;
	
		// GPU Memory
		DeviceMemory								memory;
		Vector2ul*									dAccRanges;
		LeafData*									dLeafList;	

		friend class								GPUAccLinearBatch<PGroup>;
		
	protected:

	public:
		// Constructors & Destructor
										GPUAccLinearGroup(const GPUPrimitiveGroupI&,
														  const TransformStruct*);
										~GPUAccLinearGroup() = default;

		// Interface
		// Type(as string) of the accelerator group
		const char*						Type() const override;
		// Loads required data to CPU cache for
		SceneError						InitializeGroup(// Append AABBs for each surface
														std::map<uint32_t, AABB3>& aabbOut,
														// Map of hit keys for all materials
														// w.r.t matId and primitive type
														const std::map<TypeIdPair, HitKey>&,
														// List of surface/material
														// pairings that uses this accelerator type
														// and primitive type
														const std::map<uint32_t, IdPairings>& pairingList,
														double time) override;
		SceneError						ChangeTime(std::map<uint32_t, AABB3>& aabbOut,
												   // Map of hit keys for all materials
												   // w.r.t matId and primitive type
												   const std::map<TypeIdPair, HitKey>&,
												   // List of surface/material
												   // pairings that uses this accelerator type
												   // and primitive type
												   const std::map<uint32_t, IdPairings>& pairingList,
												   double time) override;

		// Surface Queries
		uint32_t						InnerId(uint32_t surfaceId) const override;

		// Batched and singular construction
		 void							ConstructAccelerator(uint32_t surface) override;
		 void							ConstructAccelerators(const std::vector<uint32_t>& surfaces) override;
		 void							DestroyAccelerator(uint32_t surface) override;
		 void							DestroyAccelerators(const std::vector<uint32_t>& surfaces) override;

		size_t							UsedGPUMemory() const override;
		size_t							UsedCPUMemory() const override;		
};

template<class PGroup>
class GPUAccLinearBatch final
	: public GPUAcceleratorBatch<GPUAccLinearGroup<PGroup>, PGroup>
	, public LinearAccelTypeName<PGroup>
{
	public:
		// Constructors & Destructor
							GPUAccLinearBatch(const GPUAcceleratorGroupI&,
											  const GPUPrimitiveGroupI&);
							~GPUAccLinearBatch() = default;

		// Interface
		// Type(as string) of the accelerator group
		const char*			Type() const override;
		// Kernel Logic
		void				Hit(// O
								HitKey* dMaterialKeys,
								PrimitiveId* dPrimitiveIds,
								HitStructPtr dHitStructs,
								// I-O													
								RayGMem* dRays,
								// Input
								const TransformId* dTransformIds,
								const RayId* dRayIds,
								const HitKey* dAcceleratorKeys,
								const uint32_t rayCount) const override;
};

class GPUBaseAcceleratorLinear final : public GPUBaseAcceleratorI
{
	public:
		static const std::string		TypeName;
	private:
		DeviceMemory					leafMemory;
		DeviceMemory					rayLocMemory;

		// GPU
		const BaseLeaf*					dLeafs;		
		uint32_t*						dPrevLocList;

		// CPU
		std::map<uint32_t, uint32_t>	innerIds;
		uint32_t						leafCount;

	protected:
	public:
		// Interface
		// Type(as string) of the accelerator group
		const char*					Type() const override;

		// Get ready for hit loop
		void						GetReady(uint32_t rayCount) override;
		// Base accelerator only points to the next accelerator key.
		// It can return invalid key,
		// which is either means data is out of bounds or ray is invalid.		
		void						Hit(// Output
										TransformId* dTransformIds,
										HitKey* dAcceleratorKeys,
										// Inputs
										const RayGMem* dRays,
										const RayId* dRayIds,
										const uint32_t rayCount) const override;

		//TODO: define params of functions
		void						Constrcut(// List of surface to transform id hit key mappings
											  const std::map<uint32_t, BaseLeaf>&) override;
		void						Reconstruct(// List of only changed surface to transform id hit key mappings
												const std::map<uint32_t, BaseLeaf>& keys) override;
};

#include "GPUAcceleratorLinear.hpp"