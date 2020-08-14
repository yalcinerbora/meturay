
#include "TracerWorks.cuh"

// Basic Tracer Work Batches
template class DirectTracerWork<ConstantMat,
                                GPUPrimitiveEmpty,
                                EmptySurfaceFromAny<GPUPrimitiveEmpty>>;

template class DirectTracerWork<ConstantMat,
                                GPUPrimitiveTriangle,
                                EmptySurfaceFromAny<GPUPrimitiveTriangle>>;

template class DirectTracerWork<BarycentricMat,
                                GPUPrimitiveTriangle,
                                BarySurfaceFromTri>;
//
template class DirectTracerWork<ConstantMat,
                                GPUPrimitiveSphere,
                                EmptySurfaceFromAny<GPUPrimitiveSphere>>;

template class DirectTracerWork<SphericalMat,
                                GPUPrimitiveSphere,
                                SphrSurfaceFromSphr>;
// ===================================================
// Path Tracer Work Batches
template class PathTracerWork<EmissiveMat,
                              GPUPrimitiveEmpty,
                              EmptySurfaceFromAny<GPUPrimitiveEmpty>>;

template class PathTracerWork<EmissiveMat,
                              GPUPrimitiveTriangle,
                              EmptySurfaceFromAny<GPUPrimitiveTriangle>>;

template class PathTracerWork<EmissiveMat,
                              GPUPrimitiveSphere,
                              EmptySurfaceFromAny<GPUPrimitiveSphere>>;
//
template class PathTracerWork<LambertMat,
                              GPUPrimitiveTriangle,
                              BasicSurfaceFromTri>;

template class PathTracerWork<LambertMat,
                              GPUPrimitiveSphere,
                              BasicSurfaceFromSphr>;
//
template class PathTracerWork<ReflectMat,
                              GPUPrimitiveTriangle,
                              BasicSurfaceFromTri>;

template class PathTracerWork<ReflectMat,
                              GPUPrimitiveSphere,
                              BasicSurfaceFromSphr>;
//
template class PathTracerWork<RefractMat,
                              GPUPrimitiveTriangle,
                              BasicSurfaceFromTri>;

template class PathTracerWork<RefractMat,
                              GPUPrimitiveSphere,
                              BasicSurfaceFromSphr>;
// ===================================================
// Path Tracer Light Work Batches
template class PathTracerLightWork<LightMatConstant,
                                   GPUPrimitiveEmpty,
                                   EmptySurfaceFromAny<GPUPrimitiveEmpty>>;
template class PathTracerLightWork<LightMatConstant,
                                   GPUPrimitiveTriangle,
                                   EmptySurfaceFromAny<GPUPrimitiveTriangle>>;
template class PathTracerLightWork<LightMatConstant,
                                   GPUPrimitiveSphere,
                                   EmptySurfaceFromAny<GPUPrimitiveSphere>>;
template class PathTracerLightWork<LightMatTextured,
                                   GPUPrimitiveTriangle,
                                   UVSurfaceFromTri>;
template class PathTracerLightWork<LightMatTextured,
                                   GPUPrimitiveSphere,
                                   UVSurfaceFromSphr>;
template class PathTracerLightWork<LightMatCube,
                                   GPUPrimitiveEmpty,
                                   EmptySurfaceFromAny<GPUPrimitiveEmpty>>;
template class PathTracerLightWork<LightMatCube,
                                   GPUPrimitiveTriangle,
                                   EmptySurfaceFromAny<GPUPrimitiveTriangle>>;
template class PathTracerLightWork<LightMatCube,
                                   GPUPrimitiveSphere,
                                   EmptySurfaceFromAny<GPUPrimitiveSphere>>;