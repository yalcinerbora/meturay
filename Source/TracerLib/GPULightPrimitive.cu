#include "GPULightPrimitive.cuh"
#include "GPUPrimitiveSphere.h"
#include "GPUPrimitiveTriangle.h"

template class GPULight<GPUPrimitiveSphere>;
template class GPULight<GPUPrimitiveTriangle>;

template class CPULightGroup<GPUPrimitiveSphere>;
template class CPULightGroup<GPUPrimitiveTriangle>;

static_assert(IsTracerClass<CPULightGroup<GPUPrimitiveSphere>>::value,
              "CPULightGroup<GPUPrimitiveSphere> is not a Light Group Class.");
static_assert(IsTracerClass<CPULightGroup<GPUPrimitiveTriangle>>::value,
              "CPULightGroup<GPUPrimitiveTriangle> is not a Light Group Class.");