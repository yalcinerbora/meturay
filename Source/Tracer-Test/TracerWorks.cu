
#include "TracerWorks.cuh"
#include "BasicMaterials.cuh"
#include "TracerLib/GPUPrimitiveTriangle.h"
#include "TracerLib/GPUPrimitiveSphere.h"
#include "SurfaceStructs.h"


// Basic Tracer Work Batches
template class BasicTracerWork<ConstantMat,
                               GPUPrimitiveTriangle,
                               EmptySurfaceFromTri>;

template class BasicTracerWork<BarycentricMat,
                               GPUPrimitiveTriangle,
                               BarySurfaceFromTri>;
//
template class BasicTracerWork<ConstantMat,
                               GPUPrimitiveSphere,
                               EmptySurfaceFromSphr>;

template class BasicTracerWork<SphericalMat,
                               GPUPrimitiveSphere,
                               SphrSurfaceFromSphr>;
// ===================================================