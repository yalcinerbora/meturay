#include "RefPGTracerWorks.cuh"

// Ref PG Tracer Work Batches
// ===================================================
// Boundary
template class RPGBoundaryWork<BoundaryMatConstant, GPUPrimitiveEmpty>;
template class RPGBoundaryWork<BoundaryMatConstant, GPUPrimitiveTriangle>;
template class RPGBoundaryWork<BoundaryMatConstant, GPUPrimitiveSphere>;

template class RPGBoundaryWork<BoundaryMatTextured, GPUPrimitiveTriangle>;
template class RPGBoundaryWork<BoundaryMatTextured, GPUPrimitiveSphere>;

template class RPGBoundaryWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>;
// ===================================================
// Path
template class RPGPathWork<LambertCMat, GPUPrimitiveTriangle>;
template class RPGPathWork<LambertCMat, GPUPrimitiveSphere>;

template class RPGPathWork<ReflectMat, GPUPrimitiveTriangle>;
template class RPGPathWork<ReflectMat, GPUPrimitiveSphere>;

template class RPGPathWork<RefractMat, GPUPrimitiveTriangle>;
template class RPGPathWork<RefractMat, GPUPrimitiveSphere>;

template class RPGPathWork<LambertMat, GPUPrimitiveTriangle>;
template class RPGPathWork<LambertMat, GPUPrimitiveSphere>;

template class RPGPathWork<UnrealMat, GPUPrimitiveTriangle>;
template class RPGPathWork<UnrealMat, GPUPrimitiveSphere>;