#pragma once

#include "RayLib/Constants.h"
#include "RayLib/BitManipulation.h"

#include "TracerLogicI.h"
#include "GPUPrimitiveI.h"
#include "AuxiliaryDataKernels.cuh"
#include "CameraKernels.cuh"

#include "RNGMemory.h"
#include "RayMemory.h"

#include <bitset>
#include <numeric>

struct TracerError;

template<class RayAuxD>
class TracerBaseLogic : public TracerBaseLogicI
{
    public:
        using RayAuxData                    = RayAuxD;

    private:
    protected:
        // Options
        HitOpts                     optsHit;
        ShadeOpts                   optsShade;
        const TracerParameters      params;
        uint32_t                    hitStructMaxSize;
        //
        // Mappings for Kernel Calls (A.K.A. Batches)
        GPUBaseAcceleratorI&        baseAccelerator;

        AcceleratorGroupList        acceleratorGroups;
        AcceleratorBatchMappings    acceleratorBatches;

        MaterialGroupList           materialGroups;
        MaterialBatchMappings       materialBatches;

        // Estimator
        GPUEventEstimatorI&         eventEstimator;

        Vector2i                    maxAccelBits;
        Vector2i                    maxMatBits;

        HitKey                      baseBoundMatKey;

    public:
        // Constructors & Destructor
                                            TracerBaseLogic(GPUBaseAcceleratorI&,
                                                            AcceleratorGroupList&&,
                                                            AcceleratorBatchMappings&&,
                                                            MaterialGroupList&&,
                                                            MaterialBatchMappings&&,
                                                            GPUEventEstimatorI&,
                                                            //
                                                            const TracerParameters& options,
                                                            uint32_t hitStructMaxSize,
                                                            const Vector2i maxMats,
                                                            const Vector2i maxAccels,
                                                            const HitKey baseBoundMatKey);
        virtual                             ~TracerBaseLogic() = default;

        // Interface
        // Interface fetching for logic
        GPUBaseAcceleratorI&                BaseAcelerator() override { return baseAccelerator; }
        const AcceleratorBatchMappings&     AcceleratorBatches() override { return acceleratorBatches; }
        const MaterialBatchMappings&        MaterialBatches() override { return materialBatches; }
        const AcceleratorGroupList&         AcceleratorGroups() override { return acceleratorGroups; }
        const MaterialGroupList&            MaterialGroups() override { return materialGroups; }
        GPUEventEstimatorI&                 EventEstimator() override { return eventEstimator; }

        // Returns bitrange of keys (should complement each other to 32-bit)
        const Vector2i                      SceneMaterialMaxBits() const override;
        const Vector2i                      SceneAcceleratorMaxBits() const override;

        const HitKey                        SceneBaseBoundMatKey() const override;

        // Options of the Hitman & Shademan
        const HitOpts&                      HitOptions() const override { return optsHit; }
        const ShadeOpts&                    ShadeOptions() const override { return optsShade; }

        // Misc
        // Retuns "sizeof(RayAux)"
        size_t                              PerRayAuxDataSize() const override { return sizeof(RayAuxData); }
        // Return mimimum size of an arbitrary struct which holds all hit results
        size_t                              HitStructSize() const override { return hitStructMaxSize; };
        // Random seed
        uint32_t                            Seed() const override { return params.seed; }
};

template<class RayAuxD>
TracerBaseLogic<RayAuxD>::TracerBaseLogic(GPUBaseAcceleratorI& baseAccelerator,
                                          AcceleratorGroupList&& ag,
                                          AcceleratorBatchMappings&& ab,
                                          MaterialGroupList&& mg,
                                          MaterialBatchMappings&& mb,
                                          GPUEventEstimatorI& e,
                                          //
                                          const TracerParameters& params,
                                          uint32_t hitStructSize,
                                          const Vector2i maxMats,
                                          const Vector2i maxAccels,
                                          const HitKey baseBoundMatKey)
    : optsShade(TracerConstants::DefaultShadeOptions)
    , optsHit(TracerConstants::DefaultHitOptions)
    , baseAccelerator(baseAccelerator)
    , acceleratorGroups(ag)
    , acceleratorBatches(ab)
    , materialGroups(mg)
    , materialBatches(mb)
    , eventEstimator(e)
    , params(params)
    , hitStructMaxSize(hitStructSize)
    , maxAccelBits(Zero2i)
    , maxMatBits(Zero2i)
    , baseBoundMatKey(baseBoundMatKey)
{
   

    // Change count to bit
    maxMatBits[0] = Utility::FindFirstSet32(maxMats[0]) + 1;
    maxMatBits[1] = Utility::FindFirstSet32(maxMats[1]) + 1;

    maxAccelBits[0] = Utility::FindFirstSet32(maxAccels[0]) + 1;
    maxAccelBits[1] = Utility::FindFirstSet32(maxAccels[1]) + 1;
}

template<class RayAuxD>
const Vector2i TracerBaseLogic<RayAuxD>::SceneMaterialMaxBits() const
{
    return maxMatBits;
}

template<class RayAuxD>
const Vector2i TracerBaseLogic<RayAuxD>::SceneAcceleratorMaxBits() const
{
    return maxAccelBits;
}

template<class RayAuxD>
const HitKey TracerBaseLogic<RayAuxD>::SceneBaseBoundMatKey() const
{
    return baseBoundMatKey;
}