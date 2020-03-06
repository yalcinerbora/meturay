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
        const TransformStruct*          dInverseTransforms;

    public:
        // Constructors & Destructor
                                        GPUAcceleratorGroup(const GPUPrimitiveGroupI&,
                                                            const TransformStruct*);
                                        ~GPUAcceleratorGroup() = default;


        const GPUPrimitiveGroupI&       PrimitiveGroup() const override;
};

template <class P>
GPUAcceleratorGroup<P>::GPUAcceleratorGroup(const GPUPrimitiveGroupI& pg,
                                            const TransformStruct* t)
    : primitiveGroup(static_cast<const P&>(pg))
    , dInverseTransforms(t)
{}

template <class P>
const GPUPrimitiveGroupI& GPUAcceleratorGroup<P>::PrimitiveGroup() const
{
    return primitiveGroup;
}