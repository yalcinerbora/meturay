#pragma once

#include "GPUMaterialI.h"
#include "MaterialFunctions.cuh"

class GPUTransformI;

//
template <class Data>
class GPUMaterialGroupD
{
    friend struct MatDataAccessor;

    protected:
        Data    dData = Data{};
};

// Striping GPU Functionality from the Material Group
// for kernel usage.
// Each material group responsible for providing these functions

template <class D, class S,
          SampleFunc<D, S> SampleF,
          EvaluateFunc<D, S> EvalF,
          EmissionFunc<D, S> EmitF,
          IsEmissiveFunc<D> IsEmitF>
class GPUMaterialGroup
    : public GPUMaterialGroupI
    , public GPUMaterialGroupD<D>
{
    public:
        //
        using Data              = typename D;
        using Surface           = typename S;

        // Static Function Inheritance
        // Device Functions
        static constexpr SampleFunc<Data, Surface>      Sample = SampleF;
        static constexpr EvaluateFunc<Data, Surface>    Evaluate = EvalF;
        static constexpr EmissionFunc<Data, Surface>    Emit = EmitF;
        static constexpr IsEmissiveFunc<Data>           IsEmissive = IsEmitF;

    private:      
    protected:
        // Designated GPU
        const CudaGPU&                  gpu;
        std::map<uint32_t, uint32_t>    innerIds;
        const GPUTransformI* const*     dTransforms;

        SceneError                      GenerateInnerIds(const NodeListing&);

    public:
        // Constructors & Destructor
                                        GPUMaterialGroup(const CudaGPU&);
        virtual                         ~GPUMaterialGroup() = default;

        TracerError                     ConstructTextureReferences() override;

        bool                            HasMaterial(uint32_t materialId) const override;
        uint32_t                        InnerId(uint32_t materialId) const override;
        const CudaGPU&                  GPU() const override;


        virtual bool                    IsSpecularGroup() const override;
        virtual void                    AttachGlobalMediumArray(const GPUMediumI* const*,
                                                                uint32_t baseMediumIndex) override;
};

template <class D, class S, 
          SampleFunc<D, S> SF,           
          EvaluateFunc<D, S> EF,
          EmissionFunc<D, S> EmF,
          IsEmissiveFunc<D> IEF>
GPUMaterialGroup<D, S, SF, EF, EmF, IEF>::GPUMaterialGroup(const CudaGPU& gpu)
    : gpu(gpu)
{}

template <class D, class S, 
          SampleFunc<D, S> SF,           
          EvaluateFunc<D, S> EF,
          EmissionFunc<D, S> EmF,
          IsEmissiveFunc<D> IEF>
SceneError GPUMaterialGroup<D, S, SF, EF, EmF, IEF>::GenerateInnerIds(const NodeListing& nodes)
{
    uint32_t i = 0;
    for(const auto& sceneNode : nodes)
    {
        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }
    }
    return SceneError::OK;
}

template <class D, class S,
          SampleFunc<D, S> SF,
          EvaluateFunc<D, S> EF,
          EmissionFunc<D, S> EmF,
          IsEmissiveFunc<D> IEF>
TracerError GPUMaterialGroup<D, S, SF, EF, EmF, IEF>::ConstructTextureReferences()
{
    return TracerError::OK;
}

template <class D, class S, 
          SampleFunc<D, S> SF,           
          EvaluateFunc<D, S> EF,
          EmissionFunc<D, S> EmF,
          IsEmissiveFunc<D> IEF>
bool GPUMaterialGroup<D, S, SF, EF, EmF, IEF>::HasMaterial(uint32_t materialId) const
{
    if(innerIds.find(materialId) != innerIds.cend())
        return true;
    return false;
}

template <class D, class S, 
          SampleFunc<D, S> SF,           
          EvaluateFunc<D, S> EF,
          EmissionFunc<D, S> EmF,
          IsEmissiveFunc<D> IEF>
uint32_t GPUMaterialGroup<D, S, SF, EF, EmF, IEF>::InnerId(uint32_t materialId) const
{
    return innerIds.at(materialId);
}

template <class D, class S,
    SampleFunc<D, S> SF,
    EvaluateFunc<D, S> EF,
    EmissionFunc<D, S> EmF,
    IsEmissiveFunc<D> IEF>
const CudaGPU& GPUMaterialGroup<D, S, SF, EF, EmF, IEF>::GPU() const
{
    return gpu;
}

template <class D, class S,
    SampleFunc<D, S> SF,
    EvaluateFunc<D, S> EF,
    EmissionFunc<D, S> EmF,
    IsEmissiveFunc<D> IEF>
bool GPUMaterialGroup<D, S, SF, EF, EmF, IEF>::IsSpecularGroup() const
{
    return false;
}

template <class D, class S,
    SampleFunc<D, S> SF,
    EvaluateFunc<D, S> EF,
    EmissionFunc<D, S> EmF,
    IsEmissiveFunc<D> IEF>
void GPUMaterialGroup<D, S, SF, EF, EmF, IEF>::AttachGlobalMediumArray(const GPUMediumI* const*,
                                                                       uint32_t)
{}

struct MatDataAccessor
{
    // Data fetch function of the primitive
    // This struct should contain all necessary data required for kernel calls
    // related to this primitive
    // I dont know any design pattern for converting from static polymorphism
    // to dynamic one. This is my solution (it is quite werid)
    template <class MaterialGroupS>
    static typename MaterialGroupS::Data Data(const MaterialGroupS& mg)
    {
        using M = typename MaterialGroupS::Data;
        return static_cast<const GPUMaterialGroupD<M>&>(mg).dData;
    }
};