#include "GPUAcceleratorLinear.cuh"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"

#include "RayLib/ObjectFuncDefinitions.h"


const char* GPUBaseAcceleratorLinear::Type() const
{
	return TypeName.c_str();
}

void GPUBaseAcceleratorLinear::Hit(// Output
								   TransformId* dTransformIds,
								   HitKey* dAcceleratorKeys,
								   // Inputs
								   const RayGMem* dRays,
								   const RayId* dRayIds,
								   const uint32_t rayCount) const
{

}

void GPUBaseAcceleratorLinear::Constrcut(// List of allocator hitkeys of surfaces
										 const std::map<uint32_t, HitKey>&,
										 // List of all Surface/Transform pairs
										 // that will be constructed
										 const std::map<uint32_t, uint32_t>&)
{

}

void GPUBaseAcceleratorLinear::Reconstruct(// List of allocator hitkeys of surfaces
										   const std::map<uint32_t, HitKey>&,
										   // List of changed Surface/Transform pairs
										   const std::map<uint32_t, uint32_t>&)
{

}

template<class PGroup>
const std::string LinearAccelTypeName<PGroup>::TypeName = std::string("Linear") + PGroup::TypeName;

const std::string GPUBaseAcceleratorLinear::TypeName = "Linear";

// Accelerator
template class GPUAccLinearGroup<GPUPrimitiveTriangle>;
template class GPUAccLinearGroup<GPUPrimitiveSphere>;
template class GPUAccLinearBatch<GPUPrimitiveTriangle>;
template class GPUAccLinearBatch<GPUPrimitiveSphere>;
