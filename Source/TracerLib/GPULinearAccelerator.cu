#include "GPULinearAccelerator.h"
#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"

// Accelerator
template class GPUAcceleratorLinear<GPUPrimitiveTriangle>;
template class GPUAcceleratorLinear<GPUPrimitiveSphere>;