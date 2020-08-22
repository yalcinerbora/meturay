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
                 size_t I, class... Tp>
        inline constexpr typename std::enable_if<I == sizeof...(Tp), ReturnType>::type
        LoopAndFind(std::tuple<Tp...>& t)
        {
            static_assert(false, "Unable to find type in tuple");
        }

        template<class CheckType, class ReturnType,
                    size_t I, class... Tp>
        inline constexpr typename std::enable_if<(I < sizeof...(Tp)), ReturnType>::type
        LoopAndFind(std::tuple<Tp...>& t)
        {
            
            using ElementType = typename std::tuple_element_t<I, TypeList<Tp...>>::type;
            using CurrentType = typename ElementType::type;
            constexpr auto SurfaceFunc = ElementType::SurfaceGeneratorFunction;
            // Accelerator Types
            if constexpr(std::is_same_v<CurrentType, CheckType>)
                return std::get<I>(t)::SurfaceGen;
            else LoopAndFind<I + 1, Tp...>(t);
        }
    }

    template<class CheckType, class ReturnType, class... Tp>
    ReturnType LoopAndFindType(std::tuple<Tp...>& tuple)
    {
        return Detail::LoopAndFind<ReturnType, CheckType, sizeof...(Tp), Tp...>(tuple);
    }
};


template <class HitD, class PrimitiveD, class LeafD,
          class SurfaceFuncGenerator,
          AcceptHitFunction<HitD, PrimitiveD, LeafD> HitF,
          LeafGenFunction<PrimitiveD, LeafD> LeafF,
          BoxGenFunction<PrimitiveD> BoxF,
          AreaGenFunction<PrimitiveD> AreaF,
          CenterGenFunction<PrimitiveD> CenterF>
class GPUPrimitiveGroup
    : public GPUPrimitiveGroupI
    , public GPUPrimitiveGroupP<PrimitiveD>
    , public SurfaceList
{
    public:
        // Type Definitions for kernel generations
        using PrimitiveData                 = PrimitiveD;
        using HitData                       = HitD;
        using LeafData                      = LeafD;
        // Function Definitions
        // Used by accelerator definitions etc.
        static constexpr auto Hit       = HitF;
        static constexpr auto Surface   = SurfF;
        static constexpr auto Leaf      = LeafF;
        static constexpr auto AABB      = BoxF;
        static constexpr auto Area      = AreaF;
        static constexpr auto Center    = CenterF;

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