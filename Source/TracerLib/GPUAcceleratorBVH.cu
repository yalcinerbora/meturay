#include "GPUAcceleratorBVH.cuh"
#include "TypeTraits.h"

#include "GPUPrimitiveTriangle.h"
#include "GPUPrimitiveSphere.h"

// Accelerator Instancing for basic primitives
template class GPUAccBVHGroup<GPUPrimitiveTriangle>;
template class GPUAccBVHGroup<GPUPrimitiveSphere>;

template class GPUAccBVHBatch<GPUPrimitiveTriangle>;
template class GPUAccBVHBatch<GPUPrimitiveSphere>;

static_assert(IsTracerClass<GPUAccBVHGroup<GPUPrimitiveTriangle>>::value,
              "GPUAccBVHBatch<GPUPrimitiveTriangle> is not a Tracer Class.");
static_assert(IsTracerClass<GPUAccBVHGroup<GPUPrimitiveSphere>>::value,
              "GPUAccBVHGroup<GPUPrimitiveSphere> is not a Tracer Class.");