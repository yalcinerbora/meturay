#include "GPULightPrimitive.cuh"
#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveTriangle.h"

template class GPULight<GPUPrimitiveSphere>;
template class GPULight<GPUPrimitiveTriangle>;

template class CPULightGroup<GPUPrimitiveSphere>;
template class CPULightGroup<GPUPrimitiveTriangle>;