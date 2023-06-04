#include "RefPGTracerWorks.cuh"

// Ref PG Tracer Work Batches
// ===================================================
// Boundary
template class RPGBoundaryWork<CPULightGroupNull>;
template class RPGBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
template class RPGBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
template class RPGBoundaryWork<CPULightGroupSkySphere>;
template class RPGBoundaryWork<CPULightGroupPoint>;
template class RPGBoundaryWork<CPULightGroupDirectional>;
template class RPGBoundaryWork<CPULightGroupSpot>;
template class RPGBoundaryWork<CPULightGroupDisk>;
template class RPGBoundaryWork<CPULightGroupRectangular>;
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

template class RPGPathWork<MetalMat, GPUPrimitiveTriangle>;
template class RPGPathWork<MetalMat, GPUPrimitiveSphere>;