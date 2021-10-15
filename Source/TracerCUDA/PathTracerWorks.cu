#include "PathTracerWorks.cuh"
// Path Tracer Work Batches
// ===================================================
// Boundary
template class PTBoundaryWork<CPULightGroupNull>;
template class PTBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
template class PTBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
template class PTBoundaryWork<CPULightGroupSkySphere>;
template class PTBoundaryWork<CPULightGroupPoint>;
template class PTBoundaryWork<CPULightGroupDirectional>;
template class PTBoundaryWork<CPULightGroupSpot>;
template class PTBoundaryWork<CPULightGroupDisk>;
template class PTBoundaryWork<CPULightGroupRectangular>;
// ===================================================
// Path Work
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