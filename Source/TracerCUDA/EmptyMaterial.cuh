#pragma once

#include "GPUMaterialP.cuh"
#include "TypeTraits.h"
#include "EmptyMaterialKC.cuh"

template <class Surface>
class EmptyMat final
    : public GPUMaterialGroup<NullData, Surface,
                              EmptyMatFuncs<Surface>>
{
    public:
        static const char*      TypeName() { return "Empty"; }

    private:
    protected:
    public:
        // Constructors & Destructor
                                EmptyMat(const CudaGPU& gpu);
                                ~EmptyMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing&, const TextureNodeMap&,
                                                const std::map<uint32_t, uint32_t>&,
                                                double, const std::string&) override {return SceneError::OK;}
        SceneError              ChangeTime(const NodeListing&, double,
                                           const std::string&) override {return SceneError::OK;}

        // Material Queries
        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }
        size_t                  UsedGPUMemory(uint32_t) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t) const override { return 0; }

        // NEE Related
        bool                    CanBeSampled() const override { return false; }

        uint8_t                 SampleStrategyCount() const override { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const override { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const override { return std::vector<uint32_t>(); }
};

template <class S>
EmptyMat<S>::EmptyMat(const CudaGPU& gpu)
    : GPUMaterialGroup<NullData, S, EmptyMatFuncs<S>>(gpu)
{}