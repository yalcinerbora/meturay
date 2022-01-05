#include "LinearBVH.cuh"

template struct LBVHNode<PointStruct>;
template struct LinearBVHGPU<PointStruct, PointDistanceFunctor>;
template class LinearBVHCPU<PointStruct, PointDistanceFunctor, GenPointAABB>;