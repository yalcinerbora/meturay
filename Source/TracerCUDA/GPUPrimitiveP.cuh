#pragma once
/**

Template Wrapper for Primitives
P tag on name is for partial implementation
to guide the primitive implementor to make creations
proper for combined templates

*/

#include "RayLib/SceneStructs.h"

#include "GPUPrimitiveI.h"
#include "AcceleratorFunctions.h"

struct PrimDataAccessor;
class GPUTransformI;

// Surface Functions are responsible for
// generating surface structures from the hit structures
// Surface structures hold various surface data (uv, normal most of the time;
// maybe tangent if a normal map is present over a surface)
//
// Surfaces are deemed by the materials
// (i.e. a material may not require uv if a texture is present)
//
// This function is provided by the "WorkBatch" class
// meaning different WorkBatch Class is generated for different
// primitive/material pairs
template <class Surface, class HitData, class PrimitiveData>
using SurfaceFunc = Surface(*)(const HitData&,
                               const GPUTransformI&,
                               PrimitiveId,
                               const PrimitiveData&);

namespace PrimitiveSurfaceFind
{
    namespace Detail
    {
        template<class CheckType, class ReturnType,
                 size_t I, class Tuple>
        inline typename std::enable_if<I == std::tuple_size<Tuple>::value, ReturnType>::type
        constexpr LoopAndFind(Tuple&&)
        {
            static_assert(I != std::tuple_size<Tuple>::value, "Unable to find type in tuple");
        }

        template<class CheckType, class ReturnType, size_t I, class Tuple>
        inline typename std::enable_if<(I < std::tuple_size<Tuple>::value), ReturnType>::type
        constexpr LoopAndFind(Tuple&& t)
        {
            using ElementType = typename std::tuple_element_t<I, Tuple>;
            using CurrentType = typename ElementType::type;
            // Accelerator Types
            if constexpr(std::is_same_v<CurrentType, CheckType>)
            {
                constexpr auto SurfaceFunc = ElementType::SurfaceGeneratorFunction;
                return SurfaceFunc;
            }
            else return LoopAndFind<CheckType, ReturnType, I + 1, Tuple>(std::forward<Tuple>(t));

            // MSVC gives warning (missing return statement)
            return nullptr;
        }
    }

    template<class CheckType, class ReturnType, class Tuple>
    ReturnType constexpr LoopAndFindType(Tuple&& tuple)
    {
        return Detail::LoopAndFind<CheckType, ReturnType, 0, Tuple>(std::forward<Tuple>(tuple));
    }
};

template <class PrimitiveD>
class GPUPrimitiveGroupP
{
    friend struct PrimDataAccessor;

    protected:
    PrimitiveD dData = PrimitiveD{};
};

template <class HitD, class PrimitiveD, class LeafD,
          class SurfaceFuncGenerator, class PrimDeviceFunctions,
          PrimTransformType TType, uint32_t PositionPerPrimitive = 1>
class GPUPrimitiveGroup
    : public GPUPrimitiveGroupI
    , public GPUPrimitiveGroupP<PrimitiveD>
    , public SurfaceFuncGenerator
{
    public:
        // Type Definitions for kernel generations
        using PrimitiveData                         = PrimitiveD;
        using HitData                               = HitD;
        using LeafData                              = LeafD;

        // Constexpr Values
        static constexpr auto TransType             = TType;
        static constexpr uint32_t PositionPerPrim   = PositionPerPrimitive;

        static_assert(std::is_same_v<decltype(&PrimDeviceFunctions::Leaf),
                                     LeafGenFunction<PrimitiveD, LeafD>>,
                      "PrimDeviceFunctions Class Member 'Leaf' does not have correct signature");
        static_assert(std::is_same_v<decltype(&PrimDeviceFunctions::AABB),
                                     BoxGenFunction<PrimitiveD>>,
                      "PrimDeviceFunctions Class Member 'AABB' does not have correct signature");
        static_assert(std::is_same_v<decltype(&PrimDeviceFunctions::Area),
                                     AreaGenFunction<PrimitiveD>>,
                      "PrimDeviceFunctions Class Member 'Area' does not have correct signature");
        static_assert(std::is_same_v<decltype(&PrimDeviceFunctions::Center),
                                     CenterGenFunction<PrimitiveD>>,
                      "PrimDeviceFunctions Class Member 'Center' does not have correct signature");
        static_assert(std::is_same_v<decltype(&PrimDeviceFunctions::SamplePosition),
                                     SamplePosFunction<PrimitiveD>>,
                      "PrimDeviceFunctions Class Member 'SamplePosition' does not have correct signature");
        static_assert(std::is_same_v<decltype(&PrimDeviceFunctions::PositionPdfFromReference),
                                     PDFPosRefFunction<PrimitiveD>>,
                      "PrimDeviceFunctions Class Member 'PositionPDFFromReference' does not have correct signature");
        static_assert(std::is_same_v<decltype(&PrimDeviceFunctions::PositionPdfFromHit),
                                     PDFPosHitFunction<PrimitiveD>>,
                      "PrimDeviceFunctions Class Member 'PositionPDFFromHit' does not have correct signature");
        static_assert(std::is_same_v<decltype(&PrimDeviceFunctions::AcquirePositions),
                                              AcquirePosFunction<PrimitiveD, PositionPerPrim>>,
                      "PrimDeviceFunctions Class Member 'AcquirePositions' does not have correct signature");
        static_assert(std::is_same_v<decltype(&PrimDeviceFunctions::AlphaTest),
                                              AlphaTestFunction<HitD, PrimitiveD, LeafD>>,
                      "PrimDeviceFunctions Class Member 'AlphaTest' does not have correct signature");
        static_assert(std::is_same_v<decltype(&PrimDeviceFunctions::Intersects),
                                              IntersectsFunction<HitD, PrimitiveD, LeafD>>,
                      "PrimDeviceFunctions Class Member 'Intersects' does not have correct signature");

        // Function Definitions
        // Used by accelerator definitions etc.
        static constexpr auto Leaf              = PrimDeviceFunctions::Leaf;
        static constexpr auto AABB              = PrimDeviceFunctions::AABB;
        static constexpr auto Area              = PrimDeviceFunctions::Area;
        static constexpr auto Center            = PrimDeviceFunctions::Center;
        static constexpr auto SamplePosition    = PrimDeviceFunctions::SamplePosition;
        static constexpr auto PositionPdfRef    = PrimDeviceFunctions::PositionPdfFromReference;
        static constexpr auto PositionPdfHit    = PrimDeviceFunctions::PositionPdfFromHit;
        static constexpr auto AcquirePositions  = PrimDeviceFunctions::AcquirePositions;
        static constexpr auto AlphaTest         = PrimDeviceFunctions::AlphaTest;
        static constexpr auto Intersects        = PrimDeviceFunctions::Intersects;

        // Templated Intersects Function
        // This is used by OptiX instead of virtual functions
        template <class GPUTransform>
        __device__ __forceinline__
        static bool IntersectsT(float&, HitData&, const RayReg&,
                                const GPUTransform&, const LeafData&,
                                const PrimitiveData&);

    private:
    protected:
    public:
        // Constructors & Destructor
                            GPUPrimitiveGroup() = default;
        virtual             ~GPUPrimitiveGroup() = default;

        uint32_t            PrimitiveHitSize() const override { return sizeof(HitData); };
        // Most primitives are intersectable
        // Derived classes that are not intersectable should override this
        bool                IsIntersectable() const override { return true; }
        bool                IsTriangle() const override { return false; }
        PrimTransformType   TransformType() const override { return TType; };
};

struct PrimDataAccessor
{
    // Data fetch function of the primitive
    // This struct should contain all necessary data required for kernel calls
    // related to this primitive
    // I dont know any design pattern for converting from static polymorphism
    // to dynamic one. This is my solution (it is quite werid)
    template <class PrimitiveGroupS>
    static typename PrimitiveGroupS::PrimitiveData Data(const PrimitiveGroupS& pg)
    {
        using P = typename PrimitiveGroupS::PrimitiveData;
        return static_cast<const GPUPrimitiveGroupP<P>&>(pg).dData;
    }
};

template <class HitD, class PrimitiveD, class LeafD,
          class SurfaceFuncGenerator, class PrimDeviceFunctions,
          PrimTransformType TType, uint32_t PositionPerPrimitive>
template <class GPUTransform>
__device__ __forceinline__
bool GPUPrimitiveGroup<HitD, PrimitiveD, LeafD,
                       SurfaceFuncGenerator,
                       PrimDeviceFunctions,
                       TType, PositionPerPrimitive>
::IntersectsT(float& a, HitData& b, const RayReg& c,
              const GPUTransform& d, const LeafData& e,
              const PrimitiveData& f)
{
    return PrimDeviceFunctions::IntersectsT(a, b, c, d, e, f);
}