#include "LinearBVH.cuh"

template struct LBVHNode<Vector3>;
template struct LinearBVHGPU<Vector3f, DistancePoint>;
template class LinearBVHCPU<Vector3, GenPointAABB, DistancePoint>;