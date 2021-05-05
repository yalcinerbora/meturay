#include "AOTracerWork.cuh"

// ===================================================
// Ambient Occlusion Work Batches
template class AmbientOcclusionWork<GPUPrimitiveTriangle>;
template class AmbientOcclusionWork<GPUPrimitiveSphere>;