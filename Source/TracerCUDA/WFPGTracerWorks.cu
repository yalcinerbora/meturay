#include "WFPGTracerWorks.cuh"
// WFPG Tracer Work Batches
// ===================================================
// Boundary
template class WFPGBoundaryWork<CPULightGroupNull>;
template class WFPGBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
template class WFPGBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
template class WFPGBoundaryWork<CPULightGroupSkySphere>;
template class WFPGBoundaryWork<CPULightGroupPoint>;
template class WFPGBoundaryWork<CPULightGroupDirectional>;
template class WFPGBoundaryWork<CPULightGroupSpot>;
template class WFPGBoundaryWork<CPULightGroupDisk>;
template class WFPGBoundaryWork<CPULightGroupRectangular>;
// Debug Boundary
template class WFPGDebugBoundaryWork<CPULightGroup<GPUPrimitiveTriangle>>;
template class WFPGDebugBoundaryWork<CPULightGroup<GPUPrimitiveSphere>>;
template class WFPGDebugBoundaryWork<CPULightGroupSkySphere>;
template class WFPGDebugBoundaryWork<CPULightGroupPoint>;
template class WFPGDebugBoundaryWork<CPULightGroupDirectional>;
template class WFPGDebugBoundaryWork<CPULightGroupSpot>;
template class WFPGDebugBoundaryWork<CPULightGroupDisk>;
template class WFPGDebugBoundaryWork<CPULightGroupRectangular>;
template class WFPGDebugBoundaryWork<CPULightGroupConstant>;
// ===================================================
// Combo
template class WFPGWork<LambertCMat, GPUPrimitiveTriangle>;
template class WFPGWork<LambertCMat, GPUPrimitiveSphere>;

template class WFPGWork<ReflectMat, GPUPrimitiveTriangle>;
template class WFPGWork<ReflectMat, GPUPrimitiveSphere>;

template class WFPGWork<RefractMat, GPUPrimitiveTriangle>;
template class WFPGWork<RefractMat, GPUPrimitiveSphere>;

template class WFPGWork<LambertMat, GPUPrimitiveTriangle>;
template class WFPGWork<LambertMat, GPUPrimitiveSphere>;

template class WFPGWork<UnrealMat, GPUPrimitiveTriangle>;
template class WFPGWork<UnrealMat, GPUPrimitiveSphere>;
// Debug Path
template class WFPGDebugWork<LambertCMat, GPUPrimitiveTriangle>;
template class WFPGDebugWork<LambertCMat, GPUPrimitiveSphere>;

template class WFPGDebugWork<ReflectMat, GPUPrimitiveTriangle>;
template class WFPGDebugWork<ReflectMat, GPUPrimitiveSphere>;

template class WFPGDebugWork<RefractMat, GPUPrimitiveTriangle>;
template class WFPGDebugWork<RefractMat, GPUPrimitiveSphere>;

template class WFPGDebugWork<LambertMat, GPUPrimitiveTriangle>;
template class WFPGDebugWork<LambertMat, GPUPrimitiveSphere>;

template class WFPGDebugWork<UnrealMat, GPUPrimitiveTriangle>;
template class WFPGDebugWork<UnrealMat, GPUPrimitiveSphere>;