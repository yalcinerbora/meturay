
#include "TracerWorks.cuh"

// Basic Tracer Work Batches
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