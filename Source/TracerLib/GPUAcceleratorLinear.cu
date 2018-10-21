#include "GPUAcceleratorLinear.h"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"

#include "RayLib/ObjectFuncDefinitions.h"

const std::string LinearAccelTypeName::TypeName = "Linear";

// Accelerator
template class GPUAccLinearGroup<GPUPrimitiveTriangle>;
template class GPUAccLinearGroup<GPUPrimitiveSphere>;

template class GPUAccLinearBatch<GPUPrimitiveTriangle,
								 GPUAccLinearGroup<GPUPrimitiveTriangle>>;
template class GPUAccLinearBatch<GPUPrimitiveSphere,
								 GPUAccLinearGroup<GPUPrimitiveSphere>>;