#include "PPGTracerWork.cuh"
// Path Tracer Work Batches
// ===================================================
// Boundary
template class PPGBoundaryWork<CPULightGroupNull>;
template class PPGBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
template class PPGBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
template class PPGBoundaryWork<CPULightGroupSkySphere>;
template class PPGBoundaryWork<CPULightGroupPoint>;
template class PPGBoundaryWork<CPULightGroupDirectional>;
template class PPGBoundaryWork<CPULightGroupSpot>;
template class PPGBoundaryWork<CPULightGroupDisk>;
template class PPGBoundaryWork<CPULightGroupRectangular>;
// ===================================================
// Combo
template class PPGWork<LambertCMat, GPUPrimitiveTriangle>;
template class PPGWork<LambertCMat, GPUPrimitiveSphere>;

template class PPGWork<ReflectMat, GPUPrimitiveTriangle>;
template class PPGWork<ReflectMat, GPUPrimitiveSphere>;

template class PPGWork<RefractMat, GPUPrimitiveTriangle>;
template class PPGWork<RefractMat, GPUPrimitiveSphere>;

template class PPGWork<LambertMat, GPUPrimitiveTriangle>;
template class PPGWork<LambertMat, GPUPrimitiveSphere>;

template class PPGWork<UnrealMat, GPUPrimitiveTriangle>;
template class PPGWork<UnrealMat, GPUPrimitiveSphere>;