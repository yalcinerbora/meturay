#pragma once

#include "GPUMaterialP.cuh"
#include "TypeTraits.h"
#include "MetaMaterialFunctions.cuh"
#include "EmptyMaterialKC.cuh"

template <class Surface>
class EmptyMat final
    : public GPUMaterialGroup<NullData, Surface,
                              EmptySample<Surface>, EmptyEvaluate<Surface>,
                              PdfZero<NullData, Surface>,
                              EmitEmpty<NullData, Surface>,
                              IsEmissiveFalse<NullData>,
                              SpecularityPerfect<NullData, Surface>>
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
        SceneError              InitializeGroup(const NodeListing& materialNodes,
                                                const TextureNodeMap& textures,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override {return SceneError::OK;}
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override {return SceneError::OK;}

        // Material Queries
        size_t                  UsedGPUMemory() const override { return 0; }
        size_t                  UsedCPUMemory() const override { return 0; }
        size_t                  UsedGPUMemory(uint32_t materialId) const override { return 0; }
        size_t                  UsedCPUMemory(uint32_t materialId) const override { return 0; }

        // NEE Related
        bool                    IsBoundary() const override { return false; }
        bool                    CanBeSampled() const override { return false; }

        uint8_t                 SampleStrategyCount() const { return 0; };
        // No Texture
        uint8_t                 UsedTextureCount() const { return 0; }
        std::vector<uint32_t>   UsedTextureIds() const { return std::vector<uint32_t>(); }
};

template <class S>
EmptyMat<S>::EmptyMat(const CudaGPU& gpu)
    : GPUMaterialGroup<NullData, S,
                       EmptySample<S>, EmptyEvaluate<S>,
                       PdfZero<NullData, S>,
                       EmitEmpty<NullData, S>,
                       IsEmissiveFalse<NullData>,
                       SpecularityPerfect<NullData, S>>(gpu)
{}