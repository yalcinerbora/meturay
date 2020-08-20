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

template <class PrimitiveD>
class GPUPrimitiveGroupP
{
    friend struct PrimDataAccessor;

    protected:
        PrimitiveD dData = PrimitiveD{};
};

template <class HitD, class PrimitiveD, class LeafD,
          AcceptHitFunction<HitD, PrimitiveD, LeafD> HitF,
          SurfaceGenFunction<HitD, PrimitiveD> SurfF,
          LeafGenFunction<PrimitiveD, LeafD> LeafF,
          BoxGenFunction<PrimitiveD> BoxF,
          AreaGenFunction<PrimitiveD> AreaF,
          CenterGenFunction<PrimitiveD> CenterF>
class GPUPrimitiveGroup
    : public GPUPrimitiveGroupI
    , public GPUPrimitiveGroupP<PrimitiveD>
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