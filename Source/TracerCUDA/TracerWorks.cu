
#include "TracerWorks.cuh"

// Basic Tracer Work Batches
template class DirectTracerWork<ConstantMat, GPUPrimitiveEmpty>;
template class DirectTracerWork<ConstantMat, GPUPrimitiveTriangle>;
template class DirectTracerWork<ConstantMat, GPUPrimitiveSphere>;

template class DirectTracerWork<BarycentricMat, GPUPrimitiveTriangle>;
template class DirectTracerWork<SphericalMat, GPUPrimitiveSphere>;

template class DirectTracerWork<NormalRenderMat, GPUPrimitiveTriangle>;
template class DirectTracerWork<NormalRenderMat, GPUPrimitiveSphere>;

template class DirectTracerWork<LambertTexMat, GPUPrimitiveTriangle>;
template class DirectTracerWork<LambertTexMat, GPUPrimitiveSphere>;

template class DirectTracerWork<UnrealMat, GPUPrimitiveTriangle>;
template class DirectTracerWork<UnrealMat, GPUPrimitiveSphere>;
// ===================================================
// Path Tracer Work Batches
template class PathTracerWork<EmissiveMat, GPUPrimitiveEmpty>;
template class PathTracerWork<EmissiveMat, GPUPrimitiveTriangle>;
template class PathTracerWork<EmissiveMat, GPUPrimitiveSphere>;

template class PathTracerWork<LambertMat, GPUPrimitiveTriangle>;
template class PathTracerWork<LambertMat, GPUPrimitiveSphere>;

template class PathTracerWork<ReflectMat, GPUPrimitiveTriangle>;
template class PathTracerWork<ReflectMat, GPUPrimitiveSphere>;

template class PathTracerWork<RefractMat, GPUPrimitiveTriangle>;
template class PathTracerWork<RefractMat, GPUPrimitiveSphere>;

template class PathTracerWork<LambertTexMat, GPUPrimitiveTriangle>;
template class PathTracerWork<LambertTexMat, GPUPrimitiveSphere>;

template class PathTracerWork<UnrealMat, GPUPrimitiveTriangle>;
template class PathTracerWork<UnrealMat, GPUPrimitiveSphere>;

// ===================================================
// Path Tracer Light Work Batches
template class PathTracerLightWork<LightMatConstant, GPUPrimitiveEmpty>;
template class PathTracerLightWork<LightMatConstant, GPUPrimitiveTriangle>;
template class PathTracerLightWork<LightMatConstant, GPUPrimitiveSphere>;

template class PathTracerLightWork<LightMatTextured, GPUPrimitiveTriangle>;
template class PathTracerLightWork<LightMatTextured, GPUPrimitiveSphere>;

template class PathTracerLightWork<LightMatCube, GPUPrimitiveEmpty>;
template class PathTracerLightWork<LightMatCube, GPUPrimitiveTriangle>;
template class PathTracerLightWork<LightMatCube, GPUPrimitiveSphere>;