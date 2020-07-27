
#include "TracerWorks.cuh"

// Basic Tracer Work Batches
template class DirectTracerWork<ConstantMat,
                                GPUPrimitiveEmpty,
                                EmptySurfaceFromEmpty>;

template class DirectTracerWork<ConstantMat,
                                GPUPrimitiveTriangle,
                                EmptySurfaceFromTri>;

template class DirectTracerWork<BarycentricMat,
                                GPUPrimitiveTriangle,
                                BarySurfaceFromTri>;
//
template class DirectTracerWork<ConstantMat,
                                GPUPrimitiveSphere,
                                EmptySurfaceFromSphr>;

template class DirectTracerWork<SphericalMat,
                                GPUPrimitiveSphere,
                                SphrSurfaceFromSphr>;
// ===================================================
// Path Tracer Work Batches
template class PathTracerWork<EmissiveMat,
                              GPUPrimitiveEmpty,
                              EmptySurfaceFromEmpty>;

template class PathTracerWork<EmissiveMat,
                              GPUPrimitiveTriangle,
                              EmptySurfaceFromTri>;

template class PathTracerWork<EmissiveMat,
                              GPUPrimitiveSphere,
                              EmptySurfaceFromSphr>;
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
                                   EmptySurfaceFromEmpty>;
template class PathTracerLightWork<LightMatConstant,
                                   GPUPrimitiveTriangle,
                                   EmptySurfaceFromTri>;
template class PathTracerLightWork<LightMatConstant,
                                   GPUPrimitiveSphere,
                                   EmptySurfaceFromSphr>;
template class PathTracerLightWork<LightMatTextured,
                                   GPUPrimitiveTriangle,
                                   BasicUVSurfaceFromTri>;
template class PathTracerLightWork<LightMatTextured,
                                   GPUPrimitiveSphere,
                                   BasicUVSurfaceFromSphr>;
template class PathTracerLightWork<LightMatCube,
                                   GPUPrimitiveEmpty,
                                   EmptySurfaceFromEmpty>;
template class PathTracerLightWork<LightMatCube,
                                   GPUPrimitiveTriangle,
                                   EmptySurfaceFromTri>;
template class PathTracerLightWork<LightMatCube,
                                   GPUPrimitiveSphere,
                                   EmptySurfaceFromSphr>;