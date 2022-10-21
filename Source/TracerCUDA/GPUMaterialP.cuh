#pragma once

#include "GPUMaterialI.h"
#include "MaterialFunctions.h"
#include "DeviceMemory.h"
#include "GPUSurface.h"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

class GPUTransformI;

//
template <class Data>
class GPUMaterialGroupD
{
    friend struct MatDataAccessor;

    protected:
        Data    hData = Data{};

};

template <class MatType, class MatData>
__global__ static
void KCGenMaterialInerface(MatType* gConstructionLocations,
                           const GPUMaterialI** gPointerArray,
                           // Input
                           const MatData& gData,
                           uint32_t totalCount)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId >= totalCount) return;

    auto* ptr = new (gConstructionLocations + globalId) MatType(gData, globalId);
    assert(ptr == (gConstructionLocations + globalId));
    gPointerArray[globalId] = ptr;
}

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

        // Dynamic Inheritance Implementation
        class GPUMaterial final : public GPUMaterialI
        {
            private:
            const Data&             gData;  // Struct of Arrays of material data
            HitKey::Type            index;

            public:
            //
            __device__              GPUMaterial(const Data& gData, HitKey::Type index);

            // Interface
            __device__  bool        IsEmissive() const override;
            __device__  bool        Specularity(const UVSurface& surface) const override;
            __device__ Vector3f     Sample(// Sampled Output
                                           RayF& wo,                       // Out direction
                                           float& pdf,                     // PDF for Monte Carlo
                                           const GPUMediumI*& outMedium,
                                           // Input
                                           const Vector3& wi,              // Incoming Radiance
                                           const Vector3& pos,             // Position
                                           const GPUMediumI& m,
                                           //
                                           const UVSurface& surface,  // Surface info (normals uvs etc.)
                                           // I-O
                                           RNGeneratorGPUI& rng) const override;
            __device__ Vector3f     Emit(// Input
                                         const Vector3& wo,      // Outgoing Radiance
                                         const Vector3& pos,     // Position
                                         const GPUMediumI& m,
                                         //
                                         const UVSurface& surface) const override;
            __device__ Vector3f     Evaluate(// Input
                                             const Vector3& wo,              // Outgoing Radiance
                                             const Vector3& wi,              // Incoming Radiance
                                             const Vector3& pos,             // Position
                                             const GPUMediumI& m,
                                             //
                                             const UVSurface& surface) const override;

            __device__ float        Pdf(// Input
                                        const Vector3& wo,      // Outgoing Radiance
                                        const Vector3& wi,
                                        const Vector3& pos,     // Position
                                        const GPUMediumI& m,
                                        //
                                        const UVSurface& surface) const override;
        };

    private:
    protected:
        // Designated GPU
        const CudaGPU&                  gpu;
        std::map<uint32_t, uint32_t>    innerIds;
        const GPUTransformI* const*     dTransforms;
        // Dynamic Interface Related
        const GPUMaterialI**            dMaterialInterfaces;
        DeviceLocalMemory               interfaceMemory;

        // MetaSurface Generator
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

        // Dynamic Inheritance Generation
        virtual void                    GeneratePerMaterialInterfaces() override;
        virtual const GPUMaterialI**    GPUMaterialInterfaces() const override;
        virtual bool                    CanSupportDynamicInheritance() const override;
};

template <class D, class S, class DF, class P>
GPUMaterialGroupT<D, S, DF, P>::GPUMaterialGroupT(const CudaGPU& gpu)
    : gpu(gpu)
    , dTransforms(nullptr)
    , dMaterialInterfaces(nullptr)
    , interfaceMemory(&gpu)
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

template <class D, class S, class DF, class P>
void GPUMaterialGroupT<D, S, DF, P>::GeneratePerMaterialInterfaces()
{
    // Copy the SoA to Global memory
    // Allocate enough for materials
    size_t totalMatCount = innerIds.size();

    // Allocate the materials
    GPUMaterial*            dMaterials;
    const GPUMaterialI**    dMaterialInterfaces;
    Data*                   dData;
    GPUMemFuncs::AllocateMultiData(std::tie(dMaterials,
                                            dMaterialInterfaces,
                                            dData),
                                   interfaceMemory,
                                   {totalMatCount, totalMatCount, 1});

    CUDA_CHECK(cudaMemcpy(dData, &hData, sizeof(Data), cudaMemcpyHostToDevice));
    // Set the pointer
    this->dMaterialInterfaces = dMaterialInterfaces;
    // Do actual object construction (virtual pointer set etc.)
    gpu.KC_X(0,(cudaStream_t)0, totalMatCount,
             //
             KCGenMaterialInerface<GPUMaterial, Data>,
             //
             dMaterials,
             dMaterialInterfaces,
             std::ref(*dData),
             static_cast<uint32_t>(totalMatCount));
    // Interface is ready for usage!!
}

template <class D, class S, class DF, class P>
const GPUMaterialI** GPUMaterialGroupT<D, S, DF, P>::GPUMaterialInterfaces() const
{
    return dMaterialInterfaces;
}

template <class D, class S, class DF, class P>
bool GPUMaterialGroupT<D, S, DF, P>::CanSupportDynamicInheritance() const
{
    return true;
}

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
        return static_cast<const GPUMaterialGroupD<M>&>(mg).hData;
    }
};

// Dynamic Inheritance Material Class
template <class D, class S, class DF, class P>
__device__ inline
GPUMaterialGroupT<D, S, DF, P>::GPUMaterial::GPUMaterial(const Data& gData, HitKey::Type index)
    : gData(gData)
    , index(index)
{}

template <class D, class S, class DF, class P>
__device__ inline
bool GPUMaterialGroupT<D, S, DF, P>::GPUMaterial::IsEmissive() const
{
    return GPUMaterialGroupT<D, S, DF, P>::IsEmissive(gData, index);
}

template <class D, class S, class DF, class P>
__device__
bool GPUMaterialGroupT<D, S, DF, P>::GPUMaterial::Specularity(const UVSurface& surface) const
{
    static constexpr bool CAN_CONVERT_FROM_UV_SURFACE = (std::is_same_v<S, UVSurface> ||
                                                         std::is_same_v<S, BasicSurface> ||
                                                         std::is_same_v<S, EmptySurface>);

        // Don't bother SNIFAE (there are enough templates on the signature)
    // Delegate a runtime error
    if constexpr(CAN_CONVERT_FROM_UV_SURFACE)
    {
        return GPUMaterialGroupT<D, S, DF, P>::Specularity(ConvertUVSurface<S>(surface),
                                                           gData, index);
    }
    else
    {
        printf("Cuda Kernel Error: Dynamic Inheritance is used from"
               " a material that uses a surface which cannot be"
               "converted from UVSurface");
        __trap();
    }
}

template <class D, class S, class DF, class P>
__device__ inline
Vector3f GPUMaterialGroupT<D, S, DF, P>::GPUMaterial::Sample(RayF& wo,
                                                             float& pdf,
                                                             const GPUMediumI*& outMedium,
                                                             // Input
                                                             const Vector3& wi,
                                                             const Vector3& pos,
                                                             const GPUMediumI& m,
                                                             const UVSurface& surface,
                                                             RNGeneratorGPUI& rng) const
{
    static constexpr bool CAN_CONVERT_FROM_UV_SURFACE = (std::is_same_v<S, UVSurface> ||
                                                         std::is_same_v<S, BasicSurface> ||
                                                         std::is_same_v<S, EmptySurface>);
    // Don't bother SNIFAE (there are enough templates on the signature)
    // Delegate a runtime error
    if constexpr(CAN_CONVERT_FROM_UV_SURFACE)
    {
        return GPUMaterialGroupT<D, S, DF, P>::Sample(wo, pdf, outMedium, wi, pos, m,
                                                      ConvertUVSurface<S>(surface),
                                                      rng, gData, index, 0);
    }
    else
    {
        printf("Cuda Kernel Error: Dynamic Inheritance is used from"
               " a material that uses a surface which cannot be"
               "converted from UVSurface");
        __trap();
    }
}

template <class D, class S, class DF, class P>
__device__ inline
Vector3f GPUMaterialGroupT<D, S, DF, P>::GPUMaterial::Emit(// Input
                                                           const Vector3& wo,
                                                           const Vector3& pos,
                                                           const GPUMediumI& m,
                                                           //
                                                           const UVSurface& surface) const
{
    static constexpr bool CAN_CONVERT_FROM_UV_SURFACE = (std::is_same_v<S, UVSurface> ||
                                                         std::is_same_v<S, BasicSurface> ||
                                                         std::is_same_v<S, EmptySurface>);
    // Don't bother SNIFAE (there are enough templates on the signature)
    // Delegate a runtime error
    if constexpr(CAN_CONVERT_FROM_UV_SURFACE)
    {
        return GPUMaterialGroupT<D, S, DF, P>::Emit(wo, pos, m,
                                                    ConvertUVSurface<S>(surface),
                                                    gData, index);
    }
    else
    {
        printf("Cuda Kernel Error: Dynamic Inheritance is used from"
               " a material that uses a surface which cannot be"
               "converted from UVSurface");
        __trap();
    }
}


template <class D, class S, class DF, class P>
__device__ inline
Vector3f GPUMaterialGroupT<D, S, DF, P>::GPUMaterial::Evaluate(// Input
                                                               const Vector3& wo,
                                                               const Vector3& wi,
                                                               const Vector3& pos,
                                                               const GPUMediumI& m,
                                                               //
                                                               const UVSurface& surface) const
{
    static constexpr bool CAN_CONVERT_FROM_UV_SURFACE = (std::is_same_v<S, UVSurface> ||
                                                         std::is_same_v<S, BasicSurface> ||
                                                         std::is_same_v<S, EmptySurface>);
    // Don't bother SNIFAE (there are enough templates on the signature)
    // Delegate a runtime error
    if constexpr(CAN_CONVERT_FROM_UV_SURFACE)
    {
        return GPUMaterialGroupT<D, S, DF, P>::Evaluate(wo, wi, pos, m,
                                                        ConvertUVSurface<S>(surface),
                                                        gData, index);
    }
    else
    {
        printf("Cuda Kernel Error: Dynamic Inheritance is used from"
               " a material that uses a surface which cannot be"
               "converted from UVSurface");
        __trap();
    }
}

template <class D, class S, class DF, class P>
__device__ inline
float GPUMaterialGroupT<D, S, DF, P>::GPUMaterial::Pdf(// Input
                                                       const Vector3& wo,
                                                       const Vector3& wi,
                                                       const Vector3& pos,
                                                       const GPUMediumI& m,
                                                       //
                                                       const UVSurface& surface) const
{
    static constexpr bool CAN_CONVERT_FROM_UV_SURFACE = (std::is_same_v<S, UVSurface> ||
                                                         std::is_same_v<S, BasicSurface> ||
                                                         std::is_same_v<S, EmptySurface>);
    // Don't bother SNIFAE (there are enough templates on the signature)
    // Delegate a runtime error
    if constexpr(CAN_CONVERT_FROM_UV_SURFACE)
    {
        return GPUMaterialGroupT<D, S, DF, P>::Pdf(wo, wi, pos, m,
                                                   ConvertUVSurface<S>(surface),
                                                   gData, index);
    }
    else
    {
        printf("Cuda Kernel Error: Dynamic Inheritance is used from"
               " a material that uses a surface which cannot be"
               "converted from UVSurface");
        __trap();
    }
}

template <class D, class S, class DevFuncs>
using GPUMaterialGroup = GPUMaterialGroupT<D, S, DevFuncs, GPUMaterialGroupI>;