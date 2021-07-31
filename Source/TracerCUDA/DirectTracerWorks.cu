#include "DirectTracerWorks.cuh"
// ===================================================
// Direct Tracer Work Batches
template class DirectTracerFurnaceWork<BarycentricMat, GPUPrimitiveTriangle>;
template class DirectTracerFurnaceWork<SphericalMat, GPUPrimitiveSphere>;

template class DirectTracerFurnaceWork<NormalRenderMat, GPUPrimitiveTriangle>;
template class DirectTracerFurnaceWork<NormalRenderMat, GPUPrimitiveSphere>;

template class DirectTracerFurnaceWork<LambertMat, GPUPrimitiveTriangle>;
template class DirectTracerFurnaceWork<LambertMat, GPUPrimitiveSphere>;

template class DirectTracerFurnaceWork<UnrealMat, GPUPrimitiveTriangle>;
template class DirectTracerFurnaceWork<UnrealMat, GPUPrimitiveSphere>;

template class DirectTracerFurnaceWork<BoundaryMatConstant, GPUPrimitiveEmpty>;
template class DirectTracerFurnaceWork<BoundaryMatConstant, GPUPrimitiveTriangle>;
template class DirectTracerFurnaceWork<BoundaryMatConstant, GPUPrimitiveSphere>;

template class DirectTracerFurnaceWork<BoundaryMatTextured, GPUPrimitiveTriangle>;
template class DirectTracerFurnaceWork<BoundaryMatTextured, GPUPrimitiveSphere>;

template class DirectTracerFurnaceWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>;
// ===================================================
template class DirectTracerNormalWork<GPUPrimitiveEmpty>;
template class DirectTracerNormalWork<GPUPrimitiveTriangle>;
template class DirectTracerNormalWork<GPUPrimitiveSphere>;