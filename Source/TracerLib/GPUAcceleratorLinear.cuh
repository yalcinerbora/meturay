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

#include "RayLib/DeviceMemory.h"
#include "RayLib/SceneStructs.h"
#include "RayLib/Vector.h"
#include "RayLib/Constants.h"

#include "GPUAcceleratorI.h"
#include "GPUPrimitiveI.h"

#include "GPUAcceleratorLinearKC.cuh"

// This should be an array?
// Most of the time each accelerator will be constructred with a 
// Singular primitive batch, it should be better to put size constraint
//using SurfaceDataList = std::vector<uint32_t>;
using SurfaceMaterialPairs = std::array<Vector2ui, SceneConstants::MaxSurfacePerAccelerator>;

template<class PGroup>
class LinearAccelTypeName
{
	public:
		static const std::string TypeName;
};

template <class AGroup, class PGroup>
class GPUAccLinearBatch;

template <class PGroup>
class GPUAccLinearGroup final 
	: public GPUAcceleratorGroupI
	, public LinearAccelTypeName<PGroup>
{
	public:
		using LeafStruct							= PGroup::LeafStruct;

	private:		
		// From Tracer
		const PGroup&								primitiveGroup;
		const TransformStruct*						dInverseTransforms;			   		 
		// CPU Memory
		std::map<uint32_t, SurfaceMaterialPairs>	acceleratorData;

		// GPU Memory
		DeviceMemory								memory;
		const uint32_t*								dLeafCounts;
		const LeafStruct**							dLeafList;	

		friend class								GPUAccLinearBatch<GPUAccLinearGroup, PGroup>;

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
		SceneError						InitializeGroup(const std::map<uint32_t, HitKey>&,
														// List of surface nodes
														// that uses this accelerator type
														// w.r.t. this prim group
														const std::vector<SceneFileNode>&) override;

		// Batched and singular construction
		 void							ConstructAccelerator(uint32_t surface) override;
		 void							ConstructAccelerators(const std::vector<uint32_t>& surfaces) override;
		 void							DestroyAccelerator(uint32_t surface) override;
		 void							DestroyAccelerators(const std::vector<uint32_t>& surfaces) override;

		size_t							UsedGPUMemory() const override;
		size_t							UsedCPUMemory() const override;
		
		const GPUPrimitiveGroupI&		PrimitiveGroup() const override;

		
};

template <class AGroup, class PGroup>
class GPUAccLinearBatch 
	: public GPUAcceleratorBatchI
	, public LinearAccelTypeName<PGroup>
{
	private:		
		const AGroup&					acceleratorGroup;
		const PGroup&					primitiveGroup;
		
	protected:
	public:
										GPUAccLinearBatch(const GPUAcceleratorGroupI&,
														  const GPUPrimitiveGroupI&);
										~GPUAccLinearBatch() = default;
		// Type(as string) of the accelerator group
		const char*						Type() const override;
		
		void							Hit(// O
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

		// Every MaterialBatch is available for a specific primitive / accelerator data
		const GPUPrimitiveGroupI&		PrimitiveGroup() const override;
		const GPUAcceleratorGroupI&		AcceleratorGroup() const override;
};
#include "GPUAcceleratorLinear.hpp"