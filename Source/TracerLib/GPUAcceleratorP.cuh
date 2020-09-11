#pragma once
/**

..............................


*/

#include "RayLib/SceneStructs.h"

#include "GPUAcceleratorI.h"
#include "GPUPrimitiveI.h"
#include "MangledNames.h"

enum class AccTransformType
{
    CONSTANT_LOCAL_TRANSFORM,
    PER_PRIMITIVE_TRANSFORM
};

struct TransformData
{
    const AccTransformType*     gTransformTypes;
    const TransformId*          gTransformIds;
};

template <class PGroup>
class GPUAcceleratorGroup 
    :  public GPUAcceleratorGroupI
{
    private:
    protected:
        // From Tracer
        const PGroup&                   primitiveGroup;
        // Global Transform List
        const GPUTransformI**           dTransforms;

    public:
        // Constructors & Destructor
                                        GPUAcceleratorGroup(const GPUPrimitiveGroupI&);
                                        ~GPUAcceleratorGroup() = default;


        const GPUPrimitiveGroupI&       PrimitiveGroup() const override;
        void                            AttachGlobalTransformArray(const GPUTransformI** deviceTranfsorms) override;

};

template <class P>
GPUAcceleratorGroup<P>::GPUAcceleratorGroup(const GPUPrimitiveGroupI& pg)
    : primitiveGroup(static_cast<const P&>(pg))
    , dTransforms(nullptr)
{}

template <class P>
const GPUPrimitiveGroupI& GPUAcceleratorGroup<P>::PrimitiveGroup() const
{
    return primitiveGroup;
}

template <class P>
void GPUAcceleratorGroup<P>::AttachGlobalTransformArray(const GPUTransformI** deviceTranfsorms)
{
    dTransforms = deviceTranfsorms;
}