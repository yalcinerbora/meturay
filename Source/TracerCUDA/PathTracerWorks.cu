#include "PathTracerWorks.cuh"
// Path Tracer Work Batches
// ===================================================
// Boundary
template class PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveEmpty>;
template class PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveTriangle>;
template class PTBoundaryWork<BoundaryMatConstant, GPUPrimitiveSphere>;

template class PTBoundaryWork<BoundaryMatTextured, GPUPrimitiveTriangle>;
template class PTBoundaryWork<BoundaryMatTextured, GPUPrimitiveSphere>;

template class PTBoundaryWork<BoundaryMatSkySphere, GPUPrimitiveEmpty>;
// ===================================================
// Combo
template class PTPathWork<LambertCMat, GPUPrimitiveTriangle>;
template class PTPathWork<LambertCMat, GPUPrimitiveSphere>;
 
template class PTPathWork<ReflectMat, GPUPrimitiveTriangle>;
template class PTPathWork<ReflectMat, GPUPrimitiveSphere>;
 
template class PTPathWork<RefractMat, GPUPrimitiveTriangle>;
template class PTPathWork<RefractMat, GPUPrimitiveSphere>;
 
template class PTPathWork<LambertMat, GPUPrimitiveTriangle>;
template class PTPathWork<LambertMat, GPUPrimitiveSphere>;
 
template class PTPathWork<UnrealMat, GPUPrimitiveTriangle>;
template class PTPathWork<UnrealMat, GPUPrimitiveSphere>;