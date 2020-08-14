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

template <class Surface, class PGroup>
using SurfaceGenFunc = Surface(*)(const typename PGroup::PrimitiveData&,
                                  const typename PGroup::HitData&,
                                  PrimitiveId);

template <class Surface>
struct SurfaceType
{
    using type = Surface;
};

template<class S, class P>
using SurfaceGenerator = std::pair<S, SurfaceGenFunc<S, P>>;

//template <class P, class... Args>
//using SurfaceList = std::tuple<SurfaceGenerator<Args, P>...>;

template <class HitD, class PrimitiveD, class LeafD,
          AcceptHitFunction<HitD, PrimitiveD, LeafD> HitF,
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
        static constexpr auto HitFunc       = HitF;
        static constexpr auto LeafFunc      = LeafF;
        static constexpr auto BoxFunc       = BoxF;
        static constexpr auto AreaFunc      = AreaF;
        static constexpr auto CenterFunc    = CenterF;

    private:
    protected:
        
       
    public:
        // Constructors & Destructor
                            GPUPrimitiveGroup() = default;
        virtual             ~GPUPrimitiveGroup() = default;

        uint32_t            PrimitiveHitSize() const override { return sizeof(HitData); };

        template<class Surface>
        SurfaceGenFunc<Surface, GPUPrimitiveGroup>    SurfaceFunction(SurfaceType<Surface>);

};

template <class HD, class PD, class LD, AcceptHitFunction<HD, PD, LD> HF, 
          LeafGenFunction<PD, LD> LF, BoxGenFunction<PD> BF, 
          AreaGenFunction<PD> AF, CenterGenFunction<PD> CF>
template<class Surface>
SurfaceGenFunc<Surface, GPUPrimitiveGroup<HD, PD, LD, HF, LF, BF, AF, CF>> 
GPUPrimitiveGroup<HD, PD, LD, HF, LF, BF, AF, CF>::SurfaceFunction(SurfaceType<Surface> s)
{
    // Traverse the tuple


    // WHAT THE FUCK
    return nullptr;
}

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