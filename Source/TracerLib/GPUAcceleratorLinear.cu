#include "GPUAcceleratorLinear.cuh"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"

#include "RayLib/ObjectFuncDefinitions.h"

template<class PGroup>
const std::string LinearAccelTypeName<PGroup>::TypeName = std::string("Linear") + PGroup::TypeName;

// Accelerator
template class GPUAccLinearGroup<GPUPrimitiveTriangle>;
template class GPUAccLinearGroup<GPUPrimitiveSphere>;

template class GPUAccLinearBatch<GPUAccLinearGroup<GPUPrimitiveTriangle>,
								 GPUPrimitiveTriangle>;
template class GPUAccLinearBatch<GPUAccLinearGroup<GPUPrimitiveSphere>,
								 GPUPrimitiveSphere>;

