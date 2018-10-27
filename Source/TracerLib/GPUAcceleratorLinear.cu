#include "GPUAcceleratorLinear.cuh"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"

#include "RayLib/ObjectFuncDefinitions.h"


const char* GPUBaseAcceleratorLinear::Type() const
{
	return "Linear";
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

void GPUBaseAcceleratorLinear::Constrcut()
{

}

void GPUBaseAcceleratorLinear::Reconstruct()
{

}

template<class PGroup>
const std::string LinearAccelTypeName<PGroup>::TypeName = std::string("Linear") + PGroup::TypeName;

// Accelerator
template class GPUAccLinearGroup<GPUPrimitiveTriangle>;
template class GPUAccLinearGroup<GPUPrimitiveSphere>;
template class GPUAccLinearBatch<GPUPrimitiveTriangle>;
template class GPUAccLinearBatch<GPUPrimitiveSphere>;
