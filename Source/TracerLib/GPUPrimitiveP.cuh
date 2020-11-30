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
            static_assert(false, "Unable to find type in tuple");
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
          class SurfaceFuncGenerator,
          AcceptHitFunction<HitD, PrimitiveD, LeafD> HitF,
          LeafGenFunction<PrimitiveD, LeafD> LeafF,
          BoxGenFunction<PrimitiveD> BoxF,
          AreaGenFunction<PrimitiveD> AreaF,
          CenterGenFunction<PrimitiveD> CenterF,
          SampleFunction<PrimitiveD> SampleF>
class GPUPrimitiveGroup
    : public GPUPrimitiveGroupI
    , public GPUPrimitiveGroupP<PrimitiveD>
    , public SurfaceFuncGenerator
{
    public:
        // Type Definitions for kernel generations
        using PrimitiveData                 = PrimitiveD;
        using HitData                       = HitD;
        using LeafData                      = LeafD;
        // Function Definitions
        // Used by accelerator definitions etc.
        static constexpr auto Hit       = HitF;
        static constexpr auto Leaf      = LeafF;
        static constexpr auto AABB      = BoxF;
        static constexpr auto Area      = AreaF;
        static constexpr auto Center    = CenterF;
        static constexpr auto Sample    = SampleF;

    private:
    protected:
    public:
        // Constructors & Destructor
                            GPUPrimitiveGroup() = default;
        virtual             ~GPUPrimitiveGroup() = default;

        uint32_t            PrimitiveHitSize() const override { return sizeof(HitData); };
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