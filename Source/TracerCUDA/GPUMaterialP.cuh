#pragma once

#include "GPUMaterialI.h"
#include "MaterialFunctions.h"

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
          class MatDeviceFunctions,
          class Parent>
class GPUMaterialGroupT
    : public Parent
    , public GPUMaterialGroupD<D>
{
    public:
        //
        using Data              = D;
        using Surface           = S;


        static_assert(std::is_same_v<decltype(&MatDeviceFunctions::Sample),
                                     SampleFunc<Data, Surface>>,
                      "MatDeviceFunctions Class Member 'Sample' does not have correct signature");
        static_assert(std::is_same_v<decltype(&MatDeviceFunctions::Evaluate),
                                     EvaluateFunc<Data, Surface>>,
                      "MatDeviceFunctions Class Member 'Evaluate' does not have correct signature");
        static_assert(std::is_same_v<decltype(&MatDeviceFunctions::Pdf),
                                     PdfFunc<Data, Surface>>,
                      "MatDeviceFunctions Class Member 'Pdf' does not have correct signature");
        static_assert(std::is_same_v<decltype(&MatDeviceFunctions::Emit),
                                     EmissionFunc<Data, Surface>>,
                      "MatDeviceFunctions Class Member 'Emit' does not have correct signature");
        static_assert(std::is_same_v<decltype(&MatDeviceFunctions::IsEmissive),
                                     IsEmissiveFunc<Data>>,
                      "MatDeviceFunctions Class Member 'IsEmissive' does not have correct signature");
        static_assert(std::is_same_v<decltype(&MatDeviceFunctions::Specularity),
                                     SpecularityFunc<Data, Surface>>,
                      "MatDeviceFunctions Class Member 'Specularity' does not have correct signature");

        // Static Function Inheritance
        // Device Functions
        static constexpr auto Sample      = MatDeviceFunctions::Sample;
        static constexpr auto Evaluate    = MatDeviceFunctions::Evaluate;
        static constexpr auto Pdf         = MatDeviceFunctions::Pdf;
        static constexpr auto Emit        = MatDeviceFunctions::Emit;
        static constexpr auto IsEmissive  = MatDeviceFunctions::IsEmissive;
        static constexpr auto Specularity = MatDeviceFunctions::Specularity;

    private:
    protected:
        // Designated GPU
        const CudaGPU&                  gpu;
        std::map<uint32_t, uint32_t>    innerIds;
        const GPUTransformI* const*     dTransforms;

        SceneError                      GenerateInnerIds(const NodeListing&);

    public:
        // Constructors & Destructor
                                        GPUMaterialGroupT(const CudaGPU&);
        virtual                         ~GPUMaterialGroupT() = default;

        TracerError                     ConstructTextureReferences() override;

        bool                            HasMaterial(uint32_t materialId) const override;
        uint32_t                        InnerId(uint32_t materialId) const override;
        const CudaGPU&                  GPU() const override;

        virtual void                    AttachGlobalMediumArray(const GPUMediumI* const*,
                                                                uint32_t baseMediumIndex) override;
};

template <class D, class S, class DF, class P>
GPUMaterialGroupT<D, S, DF, P>::GPUMaterialGroupT(const CudaGPU& gpu)
    : gpu(gpu)
{}

template <class D, class S, class DF, class P>
SceneError GPUMaterialGroupT<D, S, DF, P>::GenerateInnerIds(const NodeListing& nodes)
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

template <class D, class S, class DF, class P>
TracerError GPUMaterialGroupT<D, S, DF, P>::ConstructTextureReferences()
{
    return TracerError::OK;
}

template <class D, class S, class DF, class P>
bool GPUMaterialGroupT<D, S, DF, P>::HasMaterial(uint32_t materialId) const
{
    if(innerIds.find(materialId) != innerIds.cend())
        return true;
    return false;
}

template <class D, class S, class DF, class P>
uint32_t GPUMaterialGroupT<D, S, DF, P>::InnerId(uint32_t materialId) const
{
    return innerIds.at(materialId);
}

template <class D, class S, class DF, class P>
const CudaGPU& GPUMaterialGroupT<D, S, DF, P>::GPU() const
{
    return gpu;
}

template <class D, class S, class DF, class P>
void GPUMaterialGroupT<D, S, DF, P>::AttachGlobalMediumArray(const GPUMediumI* const*,
                                                             uint32_t)
{}

struct MatDataAccessor
{
    // Data fetch function of the primitive
    // This struct should contain all necessary data required for kernel calls
    // related to this primitive
    // I don't know any design pattern for converting from static polymorphism
    // to dynamic one. This is my solution (it is quite weird)
    template <class MaterialGroupS>
    static typename MaterialGroupS::Data Data(const MaterialGroupS& mg)
    {
        using M = typename MaterialGroupS::Data;
        return static_cast<const GPUMaterialGroupD<M>&>(mg).dData;
    }
};

template <class D, class S, class DevFuncs>
using GPUMaterialGroup = GPUMaterialGroupT<D, S, DevFuncs, GPUMaterialGroupI>;