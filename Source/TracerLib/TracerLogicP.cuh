#pragma once

#include "RayLib/Constants.h"

#include "TracerLogicI.h"
#include "GPUPrimitiveI.h"
#include "AuxiliaryDataKernels.cuh"
#include "CameraKernels.cuh"

#include "RNGMemory.h"
#include "RayMemory.h"

#include <bitset>

struct TracerError;

template<class RayAuxD, AuxInitFunc<RayAuxD> AuxF>
class TracerBaseLogic : public TracerBaseLogicI
{
    public:
        using RayAuxData                    = RayAuxD;
        static constexpr auto AuxFunc       = AuxF;

    private:
    protected:
        // Options
        HitOpts                             optsHit;
        ShadeOpts                           optsShade;
        const TracerParameters              params;
        uint32_t                            hitStructMaxSize;
        //
        const RayAuxData                    initialValues;
        // Mappings for Kernel Calls (A.K.A. Batches)
        GPUBaseAcceleratorI&                baseAccelerator;

        AcceleratorGroupList                acceleratorGroups;
        AcceleratorBatchMappings            acceleratorBatches;

        MaterialGroupList                   materialGroups;
        MaterialBatchMappings               materialBatches;

        Vector2i                            maxAccelBits;
        Vector2i                            maxMatBits;

    public:
        // Constructors & Destructor
                                            TracerBaseLogic(GPUBaseAcceleratorI& baseAccelerator,
                                                            AcceleratorGroupList&&,
                                                            AcceleratorBatchMappings&&,
                                                            MaterialGroupList&&,
                                                            MaterialBatchMappings&&,
                                                            //
                                                            const TracerParameters& options,
                                                            const RayAuxData& initalRayAux,
                                                            uint32_t hitStructMaxSize,
                                                            const Vector2i maxMats,
                                                            const Vector2i maxAccels);
        virtual                             ~TracerBaseLogic() = default;

        // Interface
        // Interface fetching for logic
        GPUBaseAcceleratorI&                BaseAcelerator() override { return baseAccelerator; }
        const AcceleratorBatchMappings&     AcceleratorBatches() override { return acceleratorBatches; }
        const MaterialBatchMappings&        MaterialBatches() override { return materialBatches; }
        const AcceleratorGroupList&         AcceleratorGroups() override { return acceleratorGroups; }
        const MaterialGroupList&            MaterialGroups() override { return materialGroups; }

        // Returns bitrange of keys (should complement each other to 32-bit)
        const Vector2i                      SceneMaterialMaxBits() const override;
        const Vector2i                      SceneAcceleratorMaxBits() const override;

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

template<class RayAuxD, AuxInitFunc<RayAuxD> AuxF>
TracerBaseLogic<RayAuxD, AuxF>::TracerBaseLogic(GPUBaseAcceleratorI& baseAccelerator,
                                                AcceleratorGroupList&& ag,
                                                AcceleratorBatchMappings&& ab,
                                                MaterialGroupList&& mg,
                                                MaterialBatchMappings&& mb,
                                                //
                                                const TracerParameters& params,
                                                const RayAuxData& initialValues,
                                                uint32_t hitStructSize,
                                                const Vector2i maxMats,
                                                const Vector2i maxAccels)
    : optsShade(TracerConstants::DefaultShadeOptions)
    , optsHit(TracerConstants::DefaultHitOptions)
    , baseAccelerator(baseAccelerator)
    , acceleratorGroups(ag)
    , acceleratorBatches(ab)
    , materialGroups(mg)
    , materialBatches(mb)
    , params(params)
    , hitStructMaxSize(hitStructSize)
    , initialValues(initialValues)
    , maxAccelBits(Zero2i)
    , maxMatBits(Zero2i)
{
    // Change count to bit
    maxMatBits[0] = static_cast<int>(std::bitset<sizeof(int) * 8>(maxMats[0]).count());
    maxMatBits[1] = static_cast<int>(std::bitset<sizeof(int) * 8>(maxMats[1]).count());

    maxAccelBits[0] = static_cast<int>(std::bitset<sizeof(int) * 8>(maxAccels[0]).count());
    maxAccelBits[1] = static_cast<int>(std::bitset<sizeof(int) * 8>(maxAccels[1]).count());
}

template<class RayAuxD, AuxInitFunc<RayAuxD> AuxF>
const Vector2i TracerBaseLogic<RayAuxD, AuxF>::SceneMaterialMaxBits() const
{
    return maxMatBits;
}

template<class RayAuxD, AuxInitFunc<RayAuxD> AuxF>
const Vector2i TracerBaseLogic<RayAuxD, AuxF>::SceneAcceleratorMaxBits() const
{
    return maxAccelBits;
}