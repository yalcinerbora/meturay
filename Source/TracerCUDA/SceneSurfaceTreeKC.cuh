#pragma once

#include "LinearBVH.cuh"

struct SurfaceLeaf
{
    Vector3f position;
    Vector3f normal;
};

class SurfaceDistanceFunctor
{
    private:
        float               normalThreshold;
    public:
        __host__            SurfaceDistanceFunctor();
        __host__            SurfaceDistanceFunctor(float normalThreshold);
        __device__ float    operator()(const struct SurfaceLeaf& leaf,
                                       const struct SurfaceLeaf& worldSurface) const;
};

__host__ inline
SurfaceDistanceFunctor::SurfaceDistanceFunctor()
    : normalThreshold(1.0f)
{}

__host__ inline
SurfaceDistanceFunctor::SurfaceDistanceFunctor(float normalThreshold)
    : normalThreshold(normalThreshold)
{}

__device__ __forceinline__
float SurfaceDistanceFunctor::operator()(const struct SurfaceLeaf& leaf,
                                         const struct SurfaceLeaf& worldSurface) const
{
    //static constexpr float DIST_MULTIPLIER = 1.0f;
    float cosTheta = worldSurface.normal.Dot(leaf.normal);
    //float result = (worldSurface.position - leaf.position).Length();

    //if(cosTheta < normalThreshold)
    //    result *= DIST_MULTIPLIER;
    //return result;


    if(cosTheta < normalThreshold)
    {
        //printf("sn[%f, %f, %f], ln[%f, %f, %f]\n",
        //       worldSurface.normal[0],
        //       worldSurface.normal[1],
        //       worldSurface.normal[2],
        //       leaf.normal[0],
        //       leaf.normal[1],
        //       leaf.normal[2]);
        return FLT_MAX;
    }
        
    else
        return (worldSurface.position - leaf.position).Length();
    

}

__device__ __forceinline__
AABB3f GenSurfaceAABB(const SurfaceLeaf& surface)
{
    return AABB3f(surface.position, surface.position);
}


extern template struct LBVHNode<SurfaceLeaf>;
extern template struct LinearBVHGPU<SurfaceLeaf,
                                    SurfaceDistanceFunctor>;
extern template class LinearBVHCPU<SurfaceLeaf,
                                   SurfaceDistanceFunctor,
                                   GenSurfaceAABB>;

using LBVHSurfaceGPU = LinearBVHGPU<SurfaceLeaf,
                                    SurfaceDistanceFunctor>;
using LBVHSurfaceCPU = LinearBVHCPU<SurfaceLeaf,
                                    SurfaceDistanceFunctor,
                                    GenSurfaceAABB>;
