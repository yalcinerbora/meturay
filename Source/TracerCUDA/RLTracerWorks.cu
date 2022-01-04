#include "RLTracerWorks.cuh"
// RL Tracer Work Batches
// ===================================================
// Boundary
template class RLBoundaryWork<CPULightGroupNull>;
template class RLBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
template class RLBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
template class RLBoundaryWork<CPULightGroupSkySphere>;
template class RLBoundaryWork<CPULightGroupPoint>;
template class RLBoundaryWork<CPULightGroupDirectional>;
template class RLBoundaryWork<CPULightGroupSpot>;
template class RLBoundaryWork<CPULightGroupDisk>;
template class RLBoundaryWork<CPULightGroupRectangular>;
// ===================================================
// Combo
template class RLWork<LambertCMat, GPUPrimitiveTriangle>;
template class RLWork<LambertCMat, GPUPrimitiveSphere>;

template class RLWork<ReflectMat, GPUPrimitiveTriangle>;
template class RLWork<ReflectMat, GPUPrimitiveSphere>;

template class RLWork<RefractMat, GPUPrimitiveTriangle>;
template class RLWork<RefractMat, GPUPrimitiveSphere>;

template class RLWork<LambertMat, GPUPrimitiveTriangle>;
template class RLWork<LambertMat, GPUPrimitiveSphere>;

template class RLWork<UnrealMat, GPUPrimitiveTriangle>;
template class RLWork<UnrealMat, GPUPrimitiveSphere>;