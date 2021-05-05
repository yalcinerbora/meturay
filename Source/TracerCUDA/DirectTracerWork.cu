#include "DirectTracerWork.cuh"
// ===================================================
// Direct Tracer Work Batches
template class DirectTracerWork<BarycentricMat, GPUPrimitiveTriangle>;
template class DirectTracerWork<SphericalMat, GPUPrimitiveSphere>;

template class DirectTracerWork<NormalRenderMat, GPUPrimitiveTriangle>;
template class DirectTracerWork<NormalRenderMat, GPUPrimitiveSphere>;

template class DirectTracerWork<LambertMat, GPUPrimitiveTriangle>;
template class DirectTracerWork<LambertMat, GPUPrimitiveSphere>;

template class DirectTracerWork<UnrealMat, GPUPrimitiveTriangle>;
template class DirectTracerWork<UnrealMat, GPUPrimitiveSphere>;

template class DirectTracerWork<BoundaryMatConstant, GPUPrimitiveEmpty>;
template class DirectTracerWork<BoundaryMatConstant, GPUPrimitiveTriangle>;
template class DirectTracerWork<BoundaryMatConstant, GPUPrimitiveSphere>;

template class DirectTracerWork<BoundaryMatTextured, GPUPrimitiveTriangle>;
template class DirectTracerWork<BoundaryMatTextured, GPUPrimitiveSphere>;

template class DirectTracerWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>;