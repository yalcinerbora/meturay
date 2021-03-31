#pragma once
/**

..............................

*/

#include "RayLib/SceneStructs.h"

#include "GPUAcceleratorI.h"
#include "GPUPrimitiveI.h"
#include "MangledNames.h"

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
        uint32_t                        identityTransformIndex;

    public:
        // Constructors & Destructor
                                        GPUAcceleratorGroup(const GPUPrimitiveGroupI&);
                                        ~GPUAcceleratorGroup() = default;

        const GPUPrimitiveGroupI&       PrimitiveGroup() const override;
        void                            AttachGlobalTransformArray(const GPUTransformI** deviceTranfsorms,
                                                                   uint32_t identityTransformIndex) override;
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
void GPUAcceleratorGroup<P>::AttachGlobalTransformArray(const GPUTransformI** deviceTranfsorms,
                                                        uint32_t identityTIndex)
{
    dTransforms = deviceTranfsorms;
    identityTransformIndex = identityTIndex;
}