#pragma once

#include "GPUWorkI.h"
#include "RNGenerator.h"
#include "GPUPrimitiveP.cuh"
#include "GPUMaterialP.cuh"
#include "MangledNames.h"
#include "WorkKernels.cuh"
#include "GPULightNull.cuh"
#include "RNGIndependent.cuh"
#include "GPUMetaSurfaceGenerator.h"

#include "RayLib/TracerError.h"

// Meta Tracer Work Code
// With custom global Data

template <class EGroup>
__global__
static void KCGenBoundaryMetaSurfaceGenerator(GPUBoundaryMetaSurfaceGenerator<EGroup>* gInterfaceLocation,
                                              const typename EGroup::PrimitiveData& gPrimData,
                                              const typename EGroup::GPUType* gLocalLightInterfaces,
                                              const GPUTransformI* const* gTransforms,
                                              uint32_t count)
{
    using ConstructingType = GPUBoundaryMetaSurfaceGenerator<EGroup>;

    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId >= count) return;
    new (gInterfaceLocation + globalId) ConstructingType(gPrimData,
                                                         gLocalLightInterfaces,
                                                         gTransforms);
}

template <class MGroup, class PGroup>
__global__
static void KCGenMetaSurfaceGenerator(GPUMetaSurfaceGenerator<PGroup, MGroup,
                                                              PGroup::GetSurfaceFunction>* gInterfaceLocation,
                                      const typename PGroup::PrimitiveData& gPrimData,
                                      const GPUMaterialI** gLocalMaterialInterfaces,
                                      const GPUTransformI* const* gTransforms,
                                      uint32_t count)
{
    using ConstructingType = GPUMetaSurfaceGenerator<PGroup, MGroup,
                                                     PGroup::GetSurfaceFunction>;
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId >= count) return;
    new (gInterfaceLocation + globalId) ConstructingType(gPrimData,
                                                         gLocalMaterialInterfaces,
                                                         gTransforms);
}

// Material/Primitive invariant part of the code
template<class GlobalData, class RayData>
class GPUWorkBatchD : public GPUWorkBatchI
{
    protected:
        // Ray Auxiliary Input and output pointers
        // which are global (not local)
        const RayData*  dAuxInGlobal = nullptr;
        RayData*        dAuxOutLocal = nullptr;

        // GPU Friendly Struct which will be directly passed to the kernel call
        GlobalData      globalData;

    public:
        // Constructors & Destructor
                        GPUWorkBatchD() = default;
                        ~GPUWorkBatchD() = default;

        void            SetGlobalData(const GlobalData&);
        void            SetRayDataPtrs(RayData* rayDataOutLocal,
                                       const RayData* rayDataInGlobal);
};

template<class GlobalData, class RayData>
class GPUWorkBatchIntermediateI : public GPUWorkBatchD<GlobalData, RayData>
{
    public:
        virtual                             ~GPUWorkBatchIntermediateI() = default;
        // Every MaterialBatch is available for a specific primitive / material data
        virtual const GPUPrimitiveGroupI&   PrimitiveGroup() const = 0;
        virtual const GPUMaterialGroupI&    MaterialGroup() const = 0;
};

template<class GlobalData, class RayData>
class GPUWorkBatchBoundaryI : public GPUWorkBatchD<GlobalData, RayData>
{
    public:
        virtual                             ~GPUWorkBatchBoundaryI() = default;
        // Every MaterialBatch is available for a specific primitive / material data
        virtual const CPUEndpointGroupI&    EndpointGroup() const = 0;
};

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
class GPUWorkBatch
    : public GPUWorkBatchIntermediateI<GlobalData, RayData>
{
    public:
        static const char*              TypeNameGen(const char* mgOverride = nullptr,
                                                    const char* pgOverride = nullptr);
        static const char*              TypeName() {return TypeNameGen();}

    private:
        using SF =                      SurfaceFunc<typename MGroup::Surface,
                                                    typename PGroup::HitData,
                                                    typename PGroup::PrimitiveData>;
        using WF =                      WorkFunc<GlobalData, LocalData,
                                                 RayData, MGroup>;

        static constexpr SF             SurfF = SGen();
        static constexpr WF             WorkF = WFunc;

        // Meta Surface Generator
        using MetaSurfGen = GPUMetaSurfaceGenerator<PGroup, MGroup, PGroup::GetSurfaceFunction>;

    protected:
        const MGroup&                   materialGroup;
        const PGroup&                   primitiveGroup;

        const GPUTransformI* const*     dTransforms;

        // Meta Surface Generator
        // Only allocated if requested
        MetaSurfGen*                    dMetaSurfaceGenerator;
        DeviceLocalMemory               surfaceGenMemory;

        // Per-Batch Data
        LocalData                       localData;

    public:
        // Constructors & Destructor
                                        GPUWorkBatch(const GPUMaterialGroupI& mg,
                                                     const GPUPrimitiveGroupI& pg,
                                                     const GPUTransformI* const* t);
                                        ~GPUWorkBatch() = default;

        void                            Work(// Output
                                             HitKey* dBoundMatOut,
                                             RayGMem* dRayOut,
                                             //  Input
                                             const RayGMem* dRayIn,
                                             const PrimitiveId* dPrimitiveIds,
                                             const TransformId* dTransformIds,
                                             const HitStructPtr dHitStructs,
                                             // Ids
                                             const HitKey* dMatIds,
                                             const RayId* dRayIds,
                                             //
                                             const uint32_t rayCount,
                                             RNGeneratorCPUI& rngCPU) override;

        TracerError                     CreateMetaSurfaceGenerator(GPUMetaSurfaceGeneratorI*& dSurfaceGenerator) override;

        const GPUPrimitiveGroupI&       PrimitiveGroup() const override { return primitiveGroup; }
        const GPUMaterialGroupI&        MaterialGroup() const override { return materialGroup; }
};

template<class GlobalData, class LocalData,
         class RayData, class EGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, EGroup> WFunc>
class GPUBoundaryWorkBatch
    : public GPUWorkBatchBoundaryI<GlobalData, RayData>
{
    public:
        static const char*              TypeNameGen(const char* nameOverride = nullptr);
        static const char*              TypeName() {return TypeNameGen();}

    private:
        using SF =                      SurfaceFunc<typename EGroup::Surface,
                                                    typename EGroup::HitData,
                                                    typename EGroup::PrimitiveData>;
        using WF =                      BoundaryWorkFunc<GlobalData, LocalData,
                                                         RayData, EGroup>;

        static constexpr SF             SurfF = EGroup::SurfF;
        static constexpr WF             WorkF = WFunc;

        // Meta Surface Generator
        using MetaSurfGen = GPUBoundaryMetaSurfaceGenerator<EGroup>;

    protected:
        const EGroup&                   endpointGroup;

        const GPUTransformI* const*     dTransforms;

        // Per-Batch Data
        LocalData                       localData;

        // Meta Surface Generator
        // Only allocated if requested
        MetaSurfGen*                    dMetaSurfaceGenerator;
        DeviceLocalMemory               surfaceGenMemory;

    public:
        // Constructors & Destructor
                                        GPUBoundaryWorkBatch(const CPUEndpointGroupI& eg,
                                                             const GPUTransformI* const* t);
                                        ~GPUBoundaryWorkBatch() = default;

        void                            Work(// Output
                                             HitKey* dBoundMatOut,
                                             RayGMem* dRayOut,
                                             //  Input
                                             const RayGMem* dRayIn,
                                             const PrimitiveId* dPrimitiveIds,
                                             const TransformId* dTransformIds,
                                             const HitStructPtr dHitStructs,
                                             // Ids
                                             const HitKey* dMatIds,
                                             const RayId* dRayIds,
                                             //
                                             const uint32_t rayCount,
                                             RNGeneratorCPUI& rngCPU) override;

        TracerError                     CreateMetaSurfaceGenerator(GPUMetaSurfaceGeneratorI*& dSurfaceGenerator) override;

        uint8_t                         OutRayCount() const override { return 0; };
        const CPUEndpointGroupI&        EndpointGroup() const override { return endpointGroup; }
};

template<class GD, class RD>
inline void GPUWorkBatchD<GD, RD>::SetGlobalData(const GD& d)
{
    globalData = d;
}

template<class GD, class RD>
void GPUWorkBatchD<GD, RD>::SetRayDataPtrs(RD* dRDOut,
                                           const RD* dRDInGlobal)
{
    dAuxInGlobal = dRDInGlobal;
    dAuxOutLocal = dRDOut;
}

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
inline const char* GPUWorkBatch<GlobalData, LocalData, RayData,
                                MGroup, PGroup, WFunc, SGen>::TypeNameGen(const char* mgOverride,
                                                                          const char* pgOverride)
{
    const char* pgName = PGroup::TypeName();
    const char* mgName = MGroup::TypeName();
    if(pgOverride) pgName = pgOverride;
    if(mgOverride) mgName = mgOverride;

    static std::string typeName = MangledNames::WorkBatch(pgName,
                                                          mgName);
    return typeName.c_str();
}

template<class GlobalData, class LocalData, class RayData,
         class MGroup, class PGroup,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
GPUWorkBatch<GlobalData, LocalData, RayData,
             MGroup, PGroup, WFunc, SGen>::GPUWorkBatch(const GPUMaterialGroupI& mg,
                                                        const GPUPrimitiveGroupI& pg,
                                                        const GPUTransformI* const* t)
    : materialGroup(static_cast<const MGroup&>(mg))
    , primitiveGroup(static_cast<const PGroup&>(pg))
    , dTransforms(t)
    , dMetaSurfaceGenerator(nullptr)
    , surfaceGenMemory(&materialGroup.GPU())
{}

template<class GlobalData, class LocalData, class RayData, class MGroup, class PGroup,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
void GPUWorkBatch<GlobalData, LocalData, RayData,
                  MGroup, PGroup, WFunc, SGen>::Work(// Output
                                                     HitKey* dBoundMatOut,
                                                     RayGMem* dRayOut,
                                                     //  Input
                                                     const RayGMem* dRayIn,
                                                     const PrimitiveId* dPrimitiveIds,
                                                     const TransformId* dTransformIds,
                                                     const HitStructPtr dHitStructs,
                                                     // Ids
                                                     const HitKey* dMatIds,
                                                     const RayId* dRayIds,
                                                     //
                                                     const uint32_t rayCount,
                                                     RNGeneratorCPUI& rngCPU)
{
    // Do Pre-work (initialize local data etc.)
    this->GetReady();

    using PrimitiveGroup = PGroup;
    using PrimitiveData = typename PGroup::PrimitiveData;
    using HitData = typename PGroup::HitData;
    using MaterialData = typename MGroup::Data;
    using MaterialSurface = typename MGroup::Surface;

    // Get Data
    const PrimitiveData primData = PrimDataAccessor::Data(primitiveGroup);
    const MaterialData matData = MatDataAccessor::Data(materialGroup);

    const uint32_t outRayCount = this->OutRayCount();

    const CudaGPU& gpu = materialGroup.GPU();
    // TODO: Async grid stride make kernels
    // to use the same RNG
    //gpu.AsyncGridStrideKC_X
    gpu.GridStrideKC_X
    (
        0, (cudaStream_t)0,
        rayCount,
        // TODO: Generic RNG
        KCWork<GlobalData, LocalData, RayData,
               PGroup, MGroup, RNGIndependentGPU,
               WorkF, SurfF>,
        // Args
        // Output
        dBoundMatOut,
        dRayOut,
        this->dAuxOutLocal,
        outRayCount,
        // Input
        dRayIn,
        this->dAuxInGlobal,
        dPrimitiveIds,
        dTransformIds,
        dHitStructs,
        //
        dMatIds,
        dRayIds,
        // I-O
        localData,
        this->globalData,
        rngCPU.GetGPUGenerators(gpu),
        // Constants
        rayCount,
        matData,
        primData,
        dTransforms
    );
}

template<class GlobalData, class LocalData, class RayData, class MGroup, class PGroup,
         WorkFunc<GlobalData, LocalData, RayData, MGroup> WFunc,
         SurfaceFuncGenerator<typename MGroup::Surface,
                              typename PGroup::HitData,
                              typename PGroup::PrimitiveData> SGen>
TracerError GPUWorkBatch<GlobalData, LocalData, RayData,
                         MGroup, PGroup, WFunc, SGen>::CreateMetaSurfaceGenerator(GPUMetaSurfaceGeneratorI*& dSurfaceGeneratorOut)
{
    // Return if already created
    if(dMetaSurfaceGenerator)
    {
        dSurfaceGeneratorOut = dMetaSurfaceGenerator;
        return TracerError::OK;
    }

    // Generate
    if(!materialGroup.CanSupportDynamicInheritance())
        return TracerError::MATERIAL_DOES_NOT_SUPPORT_DYN_INHERITANCE;

    // Generate the material interfaces on the device memory
    const GPUMaterialI** dMaterialIPtrs = materialGroup.GPUMaterialInterfaces();

    // Acquire Primitive Data
    using PrimitiveData = typename PGroup::PrimitiveData;
    const PrimitiveData* dPrimData = primitiveGroup.GetPrimDataGPUPtr();

    // Allocate a location for this
    GPUMemFuncs::AllocateMultiData(std::tie(dMetaSurfaceGenerator),
                                   surfaceGenMemory, {1});

    // Device Construct the Class
    const auto& gpu = materialGroup.GPU();
    gpu.ExactKC_X(0, (cudaStream_t)0, 1, 1,
                  //
                  KCGenMetaSurfaceGenerator<MGroup, PGroup>,
                  //
                  dMetaSurfaceGenerator,
                  std::ref(*dPrimData),
                  dMaterialIPtrs,
                  dTransforms,
                  1);
    // Return the ptr
    dSurfaceGeneratorOut = dMetaSurfaceGenerator;
    return TracerError::OK;
}

// ======================================================= //
//                      BOUNDARY WORK                      //
// ======================================================= //

template<class GlobalData, class LocalData,
         class RayData, class EGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, EGroup> WFunc>
inline const char* GPUBoundaryWorkBatch<GlobalData, LocalData, RayData,
                                        EGroup, WFunc>::TypeNameGen(const char* nameOverride)
{
    const char* name = EGroup::TypeName();
    if(nameOverride) name = nameOverride;

    static std::string typeName = MangledNames::BoundaryWorkBatch(name);
    return typeName.c_str();
}

template<class GlobalData, class LocalData,
         class RayData, class EGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, EGroup> WFunc>
GPUBoundaryWorkBatch<GlobalData, LocalData, RayData,
                     EGroup, WFunc>::GPUBoundaryWorkBatch(const CPUEndpointGroupI& eg,
                                                          const GPUTransformI* const* t)
    : endpointGroup(static_cast<const EGroup&>(eg))
    , dTransforms(t)
    , dMetaSurfaceGenerator(nullptr)
    , surfaceGenMemory(&endpointGroup.GPU())
{}

template<class GlobalData, class LocalData,
         class RayData, class EGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, EGroup> WFunc>
void GPUBoundaryWorkBatch<GlobalData, LocalData, RayData,
                          EGroup, WFunc>::Work(// Output
                                               HitKey* dBoundMatOut,
                                               RayGMem* dRayOut,
                                               //  Input
                                               const RayGMem* dRayIn,
                                               const PrimitiveId* dPrimitiveIds,
                                               const TransformId* dTransformIds,
                                               const HitStructPtr dHitStructs,
                                               // Ids
                                               const HitKey* dMatIds,
                                               const RayId* dRayIds,
                                               //
                                               const uint32_t rayCount,
                                               RNGeneratorCPUI& rngCPU)
{
    // Special Case Skip Null Light
    if constexpr(std::is_same_v<EGroup, CPULightGroupNull>)
        return;

    // Do Pre-work (initialize local data etc.)
    this->GetReady();

    using PrimitiveGroup    = typename EGroup::PrimitiveGroup;
    using PrimitiveData     = typename EGroup::PrimitiveData;
    using HitData           = typename EGroup::HitData;
    using Surface           = typename EGroup::Surface;
    using GPUEndpointType   = typename EGroup::GPUType;

    // Get Data
    PrimitiveData primData;
    if constexpr(std::is_same_v<EGroup, CPULightGroupNull>)
        primData = {};
    else
        primData = PrimDataAccessor::Data(endpointGroup.PrimGroup());

    const GPUEndpointType* dEndpoints = endpointGroup.GPULightsDerived();

    const uint32_t outRayCount = this->OutRayCount();
    const CudaGPU& gpu = endpointGroup.GPU();

   // TODO: Async grid stride make kernels
   // to use the same RNG
   //gpu.AsyncGridStrideKC_X
    gpu.GridStrideKC_X
    (
        0, (cudaStream_t)0,
        rayCount,
        // TODO: Change This
        KCBoundaryWork<GlobalData, LocalData, RayData,
                       EGroup, RNGIndependentGPU,
                       WorkF, SurfF>,
        // Args
        // Output
        dBoundMatOut,
        dRayOut,
        this->dAuxOutLocal,
        outRayCount,
        // Input
        dRayIn,
        this->dAuxInGlobal,
        dPrimitiveIds,
        dTransformIds,
        dHitStructs,
        //
        dMatIds,
        dRayIds,
        // I-O
        localData,
        this->globalData,
        rngCPU.GetGPUGenerators(gpu),
        // Constants
        rayCount,
        dEndpoints,
        primData,
        dTransforms
    );
}

template<class GlobalData, class LocalData,
         class RayData, class EGroup,
         BoundaryWorkFunc<GlobalData, LocalData, RayData, EGroup> WFunc>
TracerError GPUBoundaryWorkBatch<GlobalData, LocalData, RayData, EGroup, WFunc>::
CreateMetaSurfaceGenerator(GPUMetaSurfaceGeneratorI*& dSurfaceGeneratorOut)
{
    // Return if already created
    if(dMetaSurfaceGenerator)
    {
        dSurfaceGeneratorOut = dMetaSurfaceGenerator;
        return TracerError::OK;
    }

    // Generate
    if(!endpointGroup.CanSupportDynamicInheritance())
        return TracerError::ENDPOINT_DOES_NOT_SUPPORT_DYN_INHERITANCE;

    using PrimitiveData = typename EGroup::PrimitiveData;
    using GPUEndpointType = typename EGroup::GPUType;

    // Acquire Primitive Data
    const PrimitiveData* dPrimData;
    if constexpr(std::is_same_v<EGroup, CPULightGroupNull>)
        dPrimData = nullptr;
    else
        dPrimData = endpointGroup.PrimGroup().GetPrimDataGPUPtr();

    // Acquire Endpoint derived class ptr list
    const GPUEndpointType* dEndpoints = endpointGroup.GPULightsDerived();

    // Allocate a location for this
    GPUMemFuncs::AllocateMultiData(std::tie(dMetaSurfaceGenerator),
                                   surfaceGenMemory, {1});

    // Device Construct the Class
    const CudaGPU& gpu = endpointGroup.GPU();
    gpu.ExactKC_X(0, (cudaStream_t)0, 1, 1,
                  //
                  KCGenBoundaryMetaSurfaceGenerator<EGroup>,
                  //
                  dMetaSurfaceGenerator,
                  std::ref(*dPrimData),
                  dEndpoints,
                  dTransforms,
                  1);
    // Return the ptr
    dSurfaceGeneratorOut = dMetaSurfaceGenerator;
    return TracerError::OK;
}