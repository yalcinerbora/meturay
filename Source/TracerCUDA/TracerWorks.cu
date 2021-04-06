#include "TracerWorks.cuh"

// Basic Tracer Work Batches
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
// ===================================================
// Path Tracer Work Batches
template class PathTracerWork<LambertCMat, GPUPrimitiveTriangle>;
template class PathTracerWork<LambertCMat, GPUPrimitiveSphere>;

template class PathTracerWork<ReflectMat, GPUPrimitiveTriangle>;
template class PathTracerWork<ReflectMat, GPUPrimitiveSphere>;

template class PathTracerWork<RefractMat, GPUPrimitiveTriangle>;
template class PathTracerWork<RefractMat, GPUPrimitiveSphere>;

template class PathTracerWork<LambertMat, GPUPrimitiveTriangle>;
template class PathTracerWork<LambertMat, GPUPrimitiveSphere>;

template class PathTracerWork<UnrealMat, GPUPrimitiveTriangle>;
template class PathTracerWork<UnrealMat, GPUPrimitiveSphere>;

template class PathTracerBoundaryWork<BoundaryMatConstant, GPUPrimitiveEmpty>;
template class PathTracerBoundaryWork<BoundaryMatConstant, GPUPrimitiveTriangle>;
template class PathTracerBoundaryWork<BoundaryMatConstant, GPUPrimitiveSphere>;

template class PathTracerBoundaryWork<BoundaryMatTextured, GPUPrimitiveTriangle>;
template class PathTracerBoundaryWork<BoundaryMatTextured, GPUPrimitiveSphere>;

template class PathTracerBoundaryWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>;
// ===================================================
// Ambient Occlusion Work Batches
template class AmbientOcclusionWork<GPUPrimitiveTriangle>;
template class AmbientOcclusionWork<GPUPrimitiveSphere>;