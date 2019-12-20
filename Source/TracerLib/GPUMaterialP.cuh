#pragma once

#include <array>

#include "GPUMaterialI.h"
#include "MaterialKernels.cuh"
#include "GPUPrimitiveP.cuh"
#include "RNGMemory.h"

struct MatDataAccessor;

template <class MaterialD>
class GPUMaterialGroupP
{
    friend struct MatDataAccessor;

    protected:
        MaterialD dData = MaterialD{};
};

// Partial Implementations
template <class TLogic, class MaterialD, class SurfaceD,
          ShadeFunc<TLogic, SurfaceD, MaterialD> ShadeF>
class GPUMaterialGroup
    : public GPUMaterialGroupI
    , public GPUMaterialGroupP<MaterialD>
{
    public:
        // Types from
        using MaterialData              = typename MaterialD;
        using Surface                   = typename SurfaceD;

        static constexpr auto ShadeFunc = ShadeF;

    private:
        const int                       gpuId;

    protected:

    public:
        // Constructors & Destructor
                                        GPUMaterialGroup(int gpuId);
        virtual                         ~GPUMaterialGroup() = default;

        int                             GPUId() const override;
};

template <class TLogic, class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SurfaceF>
class GPUMaterialBatch final : public GPUMaterialBatchI
{
    public:
        static constexpr auto           SurfFunc = SurfaceF;
        static const char*              TypeName();

    private:
        const MGroup&                   materialGroup;
        const PGroup&                   primitiveGroup;

    protected:
    public:
        // Constrcutors & Destructor
                                        GPUMaterialBatch(const GPUMaterialGroupI&,
                                                         const GPUPrimitiveGroupI&);
                                        ~GPUMaterialBatch() = default;

        // Type (as string) of the primitive group
        const char*                     Type() const override;
        // Interface
        // KC
        void                            ShadeRays(// Output
                                                  Vector4* dPixels,
                                                  //
                                                  HitKey* dBoundMatOut,
                                                  RayGMem* dRayOut,
                                                  void* dRayAuxOut,
                                                  //  Input
                                                  const RayGMem* dRayIn,
                                                  const void* dRayAuxIn,
                                                  const PrimitiveId* dPrimitiveIds,
                                                  const HitStructPtr dHitStructs,
                                                  //
                                                  const HitKey* dMatIds,
                                                  const RayId* dRayIds,
                                                  //
                                                  const uint32_t rayCount,
                                                  RNGMemory& rngMem) const override;

        const GPUPrimitiveGroupI&       PrimitiveGroup() const override;
        const GPUMaterialGroupI&        MaterialGroup() const override;

        uint8_t                         OutRayCount() const override;
};

struct MatDataAccessor
{
    // Data fetch function of the primitive
    // This struct should contain all necessary data required for kernel calls
    // related to this primitive
    // I dont know any design pattern for converting from static polymorphism
    // to dynamic one. This is my solution (it is quite werid)
    template <class MaterialGroupS>
    static typename MaterialGroupS::MaterialData Data(const MaterialGroupS& mg)
    {
        using M = typename MaterialGroupS::MaterialData;
        return static_cast<const GPUMaterialGroupP<M>&>(mg).dData;
    }
};

template <class TLogic, class MaterialD, class SurfaceD,
          ShadeFunc<TLogic, SurfaceD, MaterialD> ShadeF>
GPUMaterialGroup<TLogic, MaterialD, SurfaceD, ShadeF>::GPUMaterialGroup(int gpuId)
    : gpuId()
{}

template <class TLogic, class MaterialD, class SurfaceD,
          ShadeFunc<TLogic, SurfaceD, MaterialD> ShadeF>
int GPUMaterialGroup<TLogic, MaterialD, SurfaceD, ShadeF>::GPUId() const
{
    return gpuId;
}

template <class TLogic, class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SurfaceF>
GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::GPUMaterialBatch(const GPUMaterialGroupI& m,
                                                                     const GPUPrimitiveGroupI& p)
    : materialGroup(static_cast<const MGroup&>(m))
    , primitiveGroup(static_cast<const PGroup&>(p))
{}

template <class TLogic, class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SurfaceF>
const char* GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::TypeName()
{
   static const std::string typeName = std::string(MGroup::TypeName()) + PGroup::TypeName();
   return typeName.c_str();
}

template <class TLogic, class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SurfaceF>
const char* GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::Type() const
{
    return TypeName();
}

template <class TLogic, class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SurfaceF>
void GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::ShadeRays(// Output
                                                                   Vector4* dPixels,
                                                                   //
                                                                   HitKey* dBoundMatOut,
                                                                   RayGMem* dRayOut,
                                                                   void* dRayAuxOut,
                                                                   //  Input
                                                                   const RayGMem* dRayIn,
                                                                   const void* dRayAuxIn,
                                                                   const PrimitiveId* dPrimitiveIds,
                                                                   const HitStructPtr dHitStructs,
                                                                   //
                                                                   const HitKey* dMatIds,
                                                                   const RayId* dRayIds,

                                                                   const uint32_t rayCount,
                                                                   RNGMemory& rngMem) const
{
    using PrimitiveData = typename PGroup::PrimitiveData;
    using MaterialData = typename MGroup::MaterialData;
    using RayAuxData = typename TLogic::RayAuxData;

    // TODO: Is there a better way to implement this
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
    const MaterialData matData = MatDataAccessor::Data(materialGroup);

    const uint32_t outRayCount = materialGroup.OutRayCount();
    const int gpuId = materialGroup.GPUId();

    CudaSystem::AsyncGridStrideKC_X
    (
        gpuId,
        0,
        rayCount,
        //
        KCMaterialShade<TLogic, MGroup, PGroup, SurfFunc>,
        // Args
        // Output
        dPixels,
        //
        dBoundMatOut,
        dRayOut,
        static_cast<RayAuxData*>(dRayAuxOut),
        outRayCount,
        // Input
        dRayIn,
        static_cast<const RayAuxData*>(dRayAuxIn),
        dPrimitiveIds,
        dHitStructs,
        //
        dMatIds,
        dRayIds,
        //
        rayCount,
        rngMem.RNGData(gpuId),
        // Material Related
        matData,
        // Primitive Related
        primData
    );
}

template <class TLogic, class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SurfaceF>
const GPUPrimitiveGroupI& GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::PrimitiveGroup() const
{
    return primitiveGroup;
}

template <class TLogic, class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SurfaceF>
const GPUMaterialGroupI& GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::MaterialGroup() const
{
    return materialGroup;
}

template <class TLogic, class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SurfaceF>
uint8_t GPUMaterialBatch<TLogic, MGroup, PGroup, SurfaceF>::OutRayCount() const
{
    return materialGroup.OutRayCount();
}

//template <class TLogic, class MaterialD,
//          BoundaryShadeFunc<TLogic, MaterialD> ShadeF>
//GPUBoundaryMatGroup<TLogic, MaterialD, ShadeF>::GPUBoundaryMatGroup(int gpuId)
//    : gpuId(gpuId)
//{}
//
//template <class TLogic, class MaterialD,
//          BoundaryShadeFunc<TLogic, MaterialD> ShadeF>
//int GPUBoundaryMatGroup<TLogic, MaterialD, ShadeF>::GPUId() const
//{
//    return gpuId;
//}
//
//template <class TLogic, class MGroup>
//GPUBoundaryMatBatch<TLogic, MGroup>::GPUBoundaryMatBatch(const GPUMaterialGroupI& m,
//                                                         const GPUPrimitiveGroupI& p)
//    : materialGroup(static_cast<const MGroup&>(m))
//{}
//
//template <class TLogic, class MGroup>
//const char* GPUBoundaryMatBatch<TLogic, MGroup>::TypeName()
//{
//    static const std::string typeName(MGroup::TypeName());
//    return typeName.c_str();
//}
//
//template <class TLogic, class MGroup>
//const GPUPrimitiveGroupI* GPUBoundaryMatBatch<TLogic, MGroup>::primitiveGroup = nullptr;
//
//template <class TLogic, class MGroup>
//const char* GPUBoundaryMatBatch<TLogic, MGroup>::Type() const
//{
//    return TypeName();
//}
//
//template <class TLogic, class MGroup>
//void GPUBoundaryMatBatch<TLogic, MGroup>::ShadeRays(// Output
//                                                    Vector4* dPixels,
//                                                    //
//                                                    RayGMem* dRayOut,
//                                                    void* dRayAuxOut,
//                                                    //  Input
//                                                    const RayGMem* dRayIn,
//                                                    const void* dRayAuxIn,
//                                                    const PrimitiveId* dPrimitiveIds,
//                                                    const HitStructPtr dHitStructs,
//                                                    //
//                                                    const HitKey* dMatIds,
//                                                    const RayId* dRayIds,
//                                                    //
//                                                    const uint32_t rayCount,
//                                                    RNGMemory& rngMem) const
//{
//    using MaterialData = typename MGroup::MaterialData;
//    using RayAuxData = typename TLogic::RayAuxData;
//
//    // TODO: Is there a better way to implement this
//    const MaterialData matData = MatDataAccessor::Data(materialGroup);
//    const int gpuId = materialGroup.GPUId();
//
//    // Test
//    CudaSystem::AsyncGridStrideKC_X
//    (
//        gpuId,
//        rngMem.SharedMemorySize(StaticThreadPerBlock1D),
//        rayCount,
//        //
//        KCBoundaryMatShade<TLogic, MGroup>,
//        // Args
//        // Output
//        dPixels,
//        // Input
//        dRayIn,
//        static_cast<const RayAuxData*>(dRayAuxIn),
//        //
//        dRayIds,
//        //
//        rayCount,
//        rngMem.RNGData(gpuId),
//        // Material Related
//        matData
//    );
//}
//
//template <class TLogic, class MGroup>
//const GPUPrimitiveGroupI& GPUBoundaryMatBatch<TLogic, MGroup>::PrimitiveGroup() const
//{
//    return *primitiveGroup;
//}
//
//template <class TLogic, class MGroup>
//const GPUMaterialGroupI& GPUBoundaryMatBatch<TLogic, MGroup>::MaterialGroup() const
//{
//    return materialGroup;
//}
//
//template <class TLogic, class MGroup>
//uint8_t GPUBoundaryMatBatch<TLogic, MGroup>::OutRayCount() const
//{
//    return materialGroup.OutRayCount();
//}