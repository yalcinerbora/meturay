#pragma once

#include "RayLib/HitStructs.h"

class GPUTransformI;

class GPULocation
{
    private:      
    protected:
        // Material of the Location
        HitKey                  materialKey;
        // Transform Id of the Location
        TransformId             transformId;
        // Primitive Id of the Location (this is wrt to a primitive group)
        // These struct will be accessed with an appropirate
        PrimitiveId             primitiveId;
        // Transform class (we store transform Id in order to compare efficiently)
        const GPUTransformI&    gTransform;

    public:
        // Constructors & Destructor
        __device__                      GPULocation(HitKey, TransformId, 
                                                    PrimitiveId, 
                                                    const GPUTransformI&);
        virtual                         ~GPULocation() = default;

        // Interface
        __device__ HitKey               MaterialKey() const;
        __device__ PrimitiveId          PrimitiveIndex() const;
        __device__ TransformId          TransformIndex() const;
        __device__ const GPUTransformI& Transform() const;

        __device__ bool                 operator==(const GPULocation&) const;
        __device__ bool                 operator!=(const GPULocation&) const;
};

__device__ __forceinline__
GPULocation::GPULocation(HitKey mK, TransformId tId, 
                         PrimitiveId pId,
                         const GPUTransformI& gTrans)
    : materialKey(mK)
    , transformId(tId)
    , primitiveId(pId)
    , gTransform(gTrans)
{}

__device__ __forceinline__
HitKey GPULocation::MaterialKey() const
{
    return materialKey;
}

__device__ __forceinline__
PrimitiveId GPULocation::PrimitiveIndex() const
{
    return primitiveId;
}

__device__ __forceinline__
TransformId GPULocation::TransformIndex() const
{
    return transformId;
}

__device__ __forceinline__
const GPUTransformI& GPULocation::Transform() const
{
    return gTransform;
}

__device__ __forceinline__
bool GPULocation::operator==(const GPULocation& other) const
{
    return (transformId == other.transformId &&
            materialKey == other.materialKey &&
            transformId == other.transformId);
}

__device__ __forceinline__
bool GPULocation::operator!=(const GPULocation& other) const
{
    return !(*this == other);
}