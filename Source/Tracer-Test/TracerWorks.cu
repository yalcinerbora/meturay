
#include "TracerWorks.cuh"
#include "BasicMaterials.cuh"
#include "TracerLib/GPUPrimitiveTriangle.h"
#include "TracerLib/GPUPrimitiveSphere.h"
#include "SurfaceStructs.h"

#include "MetaWorkPool.h"

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
using BatchList = TypeList<BasicTracerWork<ConstantMat,
                                           GPUPrimitiveTriangle,
                                           EmptySurfaceFromTri>,
                           BasicTracerWork<BarycentricMat,
                                           GPUPrimitiveTriangle,
                                           BarySurfaceFromTri>,
                           BasicTracerWork<ConstantMat,
                                           GPUPrimitiveSphere,
                                           EmptySurfaceFromSphr>,
                           BasicTracerWork<SphericalMat,
                                           GPUPrimitiveSphere,
                                           SphrSurfaceFromSphr>>;

void Test()
{
    BatchList b;
    MetaWorkPool wp;
    wp.AppendGenerators(b);
}