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
// Debug Boundary
template class RLDebugBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
template class RLDebugBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
template class RLDebugBoundaryWork<CPULightGroupSkySphere>;
template class RLDebugBoundaryWork<CPULightGroupPoint>;
template class RLDebugBoundaryWork<CPULightGroupDirectional>;
template class RLDebugBoundaryWork<CPULightGroupSpot>;
template class RLDebugBoundaryWork<CPULightGroupDisk>;
template class RLDebugBoundaryWork<CPULightGroupRectangular>;
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
// Debug Path
template class RLDebugWork<LambertCMat, GPUPrimitiveTriangle>;
template class RLDebugWork<LambertCMat, GPUPrimitiveSphere>;

template class RLDebugWork<ReflectMat, GPUPrimitiveTriangle>;
template class RLDebugWork<ReflectMat, GPUPrimitiveSphere>;

template class RLDebugWork<RefractMat, GPUPrimitiveTriangle>;
template class RLDebugWork<RefractMat, GPUPrimitiveSphere>;

template class RLDebugWork<LambertMat, GPUPrimitiveTriangle>;
template class RLDebugWork<LambertMat, GPUPrimitiveSphere>;

template class RLDebugWork<UnrealMat, GPUPrimitiveTriangle>;
template class RLDebugWork<UnrealMat, GPUPrimitiveSphere>;