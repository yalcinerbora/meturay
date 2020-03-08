#pragma once

#include "GPUMaterialI.h"
#include "MaterialFunctions.cuh"

struct MatDataAccessor;

//
template <class Data>
class GPUMaterialGroupData
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
          AcquireUVList<D, S> AcqF>
class GPUMaterialGroupP
    : public GPUMaterialGroupI
    , public GPUMaterialGroupData<D>
{
    public:
        //
        using Data              = typename D;
        using Surface           = typename S;

        // Static Function Inheritance
        static constexpr SampleFunc<Data, Surface>      Sample = SampleF;
        static constexpr EvaluateFunc<Data, Surface>    Evaluate = EvalF;
        static constexpr AcquireUVList<Data, Surface>   AcquireUVList = AcqF;

    private:
        // Designated GPU
        const CudaGPU&                  gpu;
        
    protected:
    public:
        // Constructors & Destructor
                                        GPUMaterialGroupP(const CudaGPU&);
        virtual                         ~GPUMaterialGroupP() = default;

        const CudaGPU&                  GPU() const override;
};

template <class D, class S, 
          SampleFunc<D, S> SF, 
          EvaluateFunc<D, S> EF,
          AcquireUVList<D, S> AF>
GPUMaterialGroupP<D, S, SF, EF, AF>::GPUMaterialGroupP(const CudaGPU& gpu)
    : gpu(gpu)
{}

template <class D, class S, 
          SampleFunc<D, S> SF, 
          EvaluateFunc<D, S> EF,
          AcquireUVList<D, S> AF>
const CudaGPU& GPUMaterialGroupP<D, S, SF, EF, AF>::GPU() const
{
    return gpu;
}

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

//template <class TLogic, class ELogic, class MGroup, class PGroup,
//          SurfaceFunc<MGroup, PGroup> SurfaceF>
//class GPUMaterialBatch final : public GPUMaterialBatchI
//{
//    public:
//        static constexpr auto           SurfFunc = SurfaceF;
//        static const char*              TypeName();
//
//    private:
//        const MGroup&                   materialGroup;
//        const PGroup&                   primitiveGroup;
//
//    protected:
//    public:
//        // Constrcutors & Destructor
//                                        GPUMaterialBatch(const GPUMaterialGroupI&,
//                                                         const GPUPrimitiveGroupI&);
//                                        ~GPUMaterialBatch() = default;
//
//        // Type (as string) of the primitive group
//        const char*                     Type() const override;
//        // Interface
//        // KC
//        void                            ShadeRays(// Output
//                                                  ImageMemory& dImage,
//                                                  //
//                                                  HitKey* dBoundMatOut,
//                                                  RayGMem* dRayOut,
//                                                  void* dRayAuxOut,
//                                                  //  Input
//                                                  const RayGMem* dRayIn,
//                                                  const void* dRayAuxIn,
//                                                  const PrimitiveId* dPrimitiveIds,
//                                                  const HitStructPtr dHitStructs,
//                                                  //
//                                                  const HitKey* dMatIds,
//                                                  const RayId* dRayIds,                                                  
//                                                  //
//                                                  const uint32_t rayCount,
//                                                  RNGMemory& rngMem) const override;
//
//        const GPUPrimitiveGroupI&       PrimitiveGroup() const override;
//        const GPUMaterialGroupI&        MaterialGroup() const override;
//
//        uint8_t                         OutRayCount() const override;
//};
//
//template <class TLogic, class ELogic, class MaterialD, class SurfaceD,
//          ShadeFunc<TLogic, ELogic, SurfaceD, MaterialD> ShadeF>
//const GPUEventEstimatorI& GPUMaterialGroup<TLogic, ELogic, MaterialD, SurfaceD, ShadeF>::EventEstimator() const
//{
//    return estimator;
//}
//
//template <class TLogic, class ELogic, class MGroup, class PGroup,
//    SurfaceFunc<MGroup, PGroup> SurfaceF>
//    GPUMaterialBatch<TLogic, ELogic, MGroup, PGroup, SurfaceF>::GPUMaterialBatch(const GPUMaterialGroupI& m,
//                                                                                 const GPUPrimitiveGroupI& p)
//    : materialGroup(static_cast<const MGroup&>(m))
//    , primitiveGroup(static_cast<const PGroup&>(p))
//{}
//
//template <class TLogic, class ELogic, class MGroup, class PGroup,
//          SurfaceFunc<MGroup, PGroup> SurfaceF>
//const char* GPUMaterialBatch<TLogic, ELogic, MGroup, PGroup, SurfaceF>::TypeName()
//{
//    static const std::string typeName = MangledNames::MaterialBatch(TLogic::TypeName(),
//                                                                    ELogic::TypeName(),
//                                                                    PGroup::TypeName(),
//                                                                    MGroup::Name());
//    return typeName.c_str();
//}
//
//template <class TLogic, class ELogic, class MGroup, class PGroup,
//          SurfaceFunc<MGroup, PGroup> SurfaceF>
//const char* GPUMaterialBatch<TLogic, ELogic, MGroup, PGroup, SurfaceF>::Type() const
//{
//    return TypeName();
//}
//
//template <class TLogic, class ELogic, class MGroup, class PGroup,
//          SurfaceFunc<MGroup, PGroup> SurfaceF>
//void GPUMaterialBatch<TLogic, ELogic, MGroup, PGroup, SurfaceF>::ShadeRays(// Output
//                                                                           ImageMemory& dImage,
//                                                                           //
//                                                                           HitKey* dBoundMatOut,
//                                                                           RayGMem* dRayOut,
//                                                                           void* dRayAuxOut,
//                                                                           //  Input
//                                                                           const RayGMem* dRayIn,
//                                                                           const void* dRayAuxIn,
//                                                                           const PrimitiveId* dPrimitiveIds,
//                                                                           const HitStructPtr dHitStructs,
//                                                                           //
//                                                                           const HitKey* dMatIds,
//                                                                           const RayId* dRayIds,
//                                                                           //
//                                                                           const uint32_t rayCount,
//                                                                           RNGMemory& rngMem) const
//{
//    using PrimitiveData = typename PGroup::PrimitiveData;
//    using MaterialData = typename MGroup::MaterialData;
//    using RayAuxData = typename TLogic::RayAuxData;
//    using EstimatorData = typename ELogic::EstimatorData;
//    // TODO: Is there a better way to implement this
//    const ELogic& estimator = static_cast<const ELogic&>(materialGroup.EventEstimator());
//
//    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
//    const MaterialData matData = MatDataAccessor::Data(materialGroup);    
//    const EstimatorData estData = EstimatorDataAccessor::Data(estimator);
//
//    const uint32_t outRayCount = materialGroup.OutRayCount();
//    const CudaGPU& gpu = materialGroup.GPU();
//
//    gpu.AsyncGridStrideKC_X
//    (
//        0,
//        rayCount,
//        //
//        KCMaterialShade<TLogic, ELogic, MGroup, PGroup, SurfFunc>,
//        // Args
//        // Output
//        dImage.GMem<Vector4f>(),
//        //
//        dBoundMatOut,
//        dRayOut,
//        static_cast<RayAuxData*>(dRayAuxOut),
//        outRayCount,
//        // Input
//        dRayIn,
//        static_cast<const RayAuxData*>(dRayAuxIn),
//        dPrimitiveIds,
//        dHitStructs,
//        //
//        dMatIds,
//        dRayIds,
//        //
//        rayCount,
//        rngMem.RNGData(gpu),
//        // Estimator
//        estData,
//        // Material Related
//        matData,
//        // Primitive Related
//        primData
//    );
//}
//
//template <class TLogic, class ELogic, class MGroup, class PGroup,
//          SurfaceFunc<MGroup, PGroup> SurfaceF>
//const GPUPrimitiveGroupI& GPUMaterialBatch<TLogic, ELogic, MGroup, PGroup, SurfaceF>::PrimitiveGroup() const
//{
//    return primitiveGroup;
//}
//
//template <class TLogic, class ELogic, class MGroup, class PGroup,
//          SurfaceFunc<MGroup, PGroup> SurfaceF>
//const GPUMaterialGroupI& GPUMaterialBatch<TLogic, ELogic, MGroup, PGroup, SurfaceF>::MaterialGroup() const
//{
//    return materialGroup;
//}
//
//template <class TLogic, class ELogic, class MGroup, class PGroup,
//          SurfaceFunc<MGroup, PGroup> SurfaceF>
//uint8_t GPUMaterialBatch<TLogic, ELogic, MGroup, PGroup, SurfaceF>::OutRayCount() const
//{
//    return materialGroup.OutRayCount();
//}