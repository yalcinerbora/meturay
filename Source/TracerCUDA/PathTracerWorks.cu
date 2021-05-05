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
 template class PTComboWork<LambertCMat, GPUPrimitiveTriangle>;
 template class PTComboWork<LambertCMat, GPUPrimitiveSphere>;
 
 template class PTComboWork<ReflectMat, GPUPrimitiveTriangle>;
 template class PTComboWork<ReflectMat, GPUPrimitiveSphere>;
 
 template class PTComboWork<RefractMat, GPUPrimitiveTriangle>;
 template class PTComboWork<RefractMat, GPUPrimitiveSphere>;
 
 template class PTComboWork<LambertMat, GPUPrimitiveTriangle>;
 template class PTComboWork<LambertMat, GPUPrimitiveSphere>;
 
 template class PTComboWork<UnrealMat, GPUPrimitiveTriangle>;
 template class PTComboWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
// Path
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
// ===================================================
// NEE
template class PTNEEWork<LambertCMat, GPUPrimitiveTriangle>;
template class PTNEEWork<LambertCMat, GPUPrimitiveSphere>;

template class PTNEEWork<ReflectMat, GPUPrimitiveTriangle>;
template class PTNEEWork<ReflectMat, GPUPrimitiveSphere>;

template class PTNEEWork<RefractMat, GPUPrimitiveTriangle>;
template class PTNEEWork<RefractMat, GPUPrimitiveSphere>;

template class PTNEEWork<LambertMat, GPUPrimitiveTriangle>;
template class PTNEEWork<LambertMat, GPUPrimitiveSphere>;

template class PTNEEWork<UnrealMat, GPUPrimitiveTriangle>;
template class PTNEEWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
// MIS
template class PTMISWork<LambertCMat, GPUPrimitiveTriangle>;
template class PTMISWork<LambertCMat, GPUPrimitiveSphere>;

template class PTMISWork<ReflectMat, GPUPrimitiveTriangle>;
template class PTMISWork<ReflectMat, GPUPrimitiveSphere>;

template class PTMISWork<RefractMat, GPUPrimitiveTriangle>;
template class PTMISWork<RefractMat, GPUPrimitiveSphere>;

template class PTMISWork<LambertMat, GPUPrimitiveTriangle>;
template class PTMISWork<LambertMat, GPUPrimitiveSphere>;

template class PTMISWork<UnrealMat, GPUPrimitiveTriangle>;
template class PTMISWork<UnrealMat, GPUPrimitiveSphere>;