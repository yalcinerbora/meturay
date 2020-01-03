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

template <class AGroup, class PGroup>
class GPUAcceleratorBatch : public GPUAcceleratorBatchI
{
    public:
        static const char*              TypeName() { return AGroup::TypeName(); }

    private:
    protected:
        const AGroup&                   acceleratorGroup;
        const PGroup&                   primitiveGroup;

    public:
        // Constructors & Destructor
                                        GPUAcceleratorBatch(const GPUAcceleratorGroupI&,
                                                            const GPUPrimitiveGroupI&);
                                        ~GPUAcceleratorBatch() = default;

        // Every MaterialBatch is available for a specific primitive / accelerator data
        const GPUPrimitiveGroupI&       PrimitiveGroup() const override;
        const GPUAcceleratorGroupI&     AcceleratorGroup() const override;
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

template<class A, class P>
GPUAcceleratorBatch<A, P>::GPUAcceleratorBatch(const GPUAcceleratorGroupI& a,
                                               const GPUPrimitiveGroupI& p)
    : acceleratorGroup(static_cast<const A&>(a))
    , primitiveGroup(static_cast<const P&>(p))
{}

template <class A, class P>
const GPUPrimitiveGroupI& GPUAcceleratorBatch<A, P>::PrimitiveGroup() const
{
    return primitiveGroup;
}

template <class A, class P>
const GPUAcceleratorGroupI& GPUAcceleratorBatch<A, P>::AcceleratorGroup() const
{
    return acceleratorGroup;
}