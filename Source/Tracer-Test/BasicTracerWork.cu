#include "BasicTracerWork.cuh"
#include "BasicMaterials.cuh"
#include "TracerLib/GPUPrimitiveTriangle.h"
#include "TracerLib/GPUPrimitiveSphere.h"
#include "SurfaceStructs.h"

template class BasicTracerBatch<ConstantMat, 
                                GPUPrimitiveTriangle,
                                EmptySurfaceFromTri>;

template class BasicTracerBatch<BarycentricMat, 
                                GPUPrimitiveTriangle,
                                BarySurfaceFromTri>;
// ===================================================
template class BasicTracerBatch<ConstantMat, 
                                GPUPrimitiveSphere,
                                EmptySurfaceFromSphr>;

template class BasicTracerBatch<SphericalMat, 
                                GPUPrimitiveSphere,
                                SphrSurfaceFromSphr>;