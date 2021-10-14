#include "DirectTracerWorks.cuh"
// ===================================================
// Direct Tracer Boundary Work Batches
template class DirectTracerBoundaryWork<CPULightGroupNull>;
template class DirectTracerBoundaryWork<CPULightGroupSkySphere>;
// ===================================================
// Direct Tracer Work Batches
template class DirectTracerFurnaceWork<BarycentricMat, GPUPrimitiveTriangle>;
template class DirectTracerFurnaceWork<SphericalMat, GPUPrimitiveSphere>;

template class DirectTracerFurnaceWork<NormalRenderMat, GPUPrimitiveTriangle>;
template class DirectTracerFurnaceWork<NormalRenderMat, GPUPrimitiveSphere>;

template class DirectTracerFurnaceWork<LambertCMat, GPUPrimitiveTriangle>;
template class DirectTracerFurnaceWork<LambertCMat, GPUPrimitiveSphere>;

template class DirectTracerFurnaceWork<LambertMat, GPUPrimitiveTriangle>;
template class DirectTracerFurnaceWork<LambertMat, GPUPrimitiveSphere>;

template class DirectTracerFurnaceWork<UnrealMat, GPUPrimitiveTriangle>;
template class DirectTracerFurnaceWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
template class DirectTracerNormalWork<GPUPrimitiveEmpty>;
template class DirectTracerNormalWork<GPUPrimitiveTriangle>;
template class DirectTracerNormalWork<GPUPrimitiveSphere>;