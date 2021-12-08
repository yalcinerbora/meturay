#pragma once
/**

Unreal Engine Physically Based Shader Implementation

https://www.semanticscholar.org/paper/Real-Shading-in-Unreal-Engine-4-by-Karis/91ee695f6a64d8508817fa3c0203d4389c462536

Implementation of Micro-facet Shader definition of the Unreal Engine

*/

#include "UnrealMaterialKC.cuh"

#include "GPUSurface.h"
#include "GPUMaterialP.cuh"
#include "TypeTraits.h"
#include "DeviceMemory.h"
#include "Texture.cuh"

class UnrealMat final
    : public GPUMaterialGroup<UnrealMatData, UVSurface,
                              UnrealDeviceFuncs>
{
    public:
        static const char*          TypeName() { return "Unreal"; }

        using Constant1CRef = ConstantRef<2, float>;
        using Constant3CRef = ConstantRef<2, Vector3>;
        using Texture2D1CRef = TextureRef<2, float>;
        using Texture2D3CRef = TextureRef<2, Vector3>;
        using Texture2D1CRefI = TextureRefI<2, float>;
        using Texture2D3CRefI = TextureRefI<2, Vector3>;

        static constexpr int TotalParameterCount = 5;
        using TextureIdList = std::array<uint32_t, TotalParameterCount>;
        using Tex2DMap = std::map<uint32_t, std::unique_ptr<TextureI<2>>>;

        struct ConstructionInfo
        {
            // You could use union here but memory is not a concern
            // std::variant causes error on CUDA *cu files
            // (even if the code is on device functions)
            //
            bool                hasNormalMap = false;
            cudaTextureObject_t normalMap = 0;
            //
            bool                hasAlbedoMap = false;
            cudaTextureObject_t albedoMap = 0;
            Vector3f            albedoConst;
            //
            bool                hasMetallicMap = false;
            cudaTextureObject_t metallicMap = 0;
            float               metallicConst;
            //
            bool                hasSpecularMap = false;
            cudaTextureObject_t specularMap = 0;
            float               specularConst;
            //
            bool                hasRoughnessMap = false;
            cudaTextureObject_t roughnessMap = 0;
            float               roughnessConst;
        };

        enum TexType
        {
            ALBEDO,
            NORMAL,
            METALLIC,
            ROUGHNESS,
            SPECULAR
        };

    private:
        DeviceMemory            memory;
        // Actual Texture Allocations
        Tex2DMap                dTextureMemory;
        // Device Allocations of Constant References
        const Constant3CRef*    dConstAlbedo;
        const Constant1CRef*    dConstMetallic;
        const Constant1CRef*    dConstSpecular;
        const Constant1CRef*    dConstRoughness;
        // Device Allocations of Texture References
        const Texture2D3CRef*   dTextureAlbedoRef;
        const Texture2D1CRef*   dTextureMetallicRef;
        const Texture2D1CRef*   dTextureSpecularRef;
        const Texture2D1CRef*   dTextureRoughnessRef;
        const Texture2D3CRef*   dTextureNormalRef;
        // Aligned pointers for material access from kernel
        const Texture2D3CRefI** dAlbedo;
        const Texture2D1CRefI** dMetallic;
        const Texture2D1CRefI** dSpecular;
        const Texture2D1CRefI** dRoughness;
        const Texture2D3CRefI** dNormal;

        // CPU Construction Info
        std::vector<ConstructionInfo> matConstructionInfo;
        std::vector<TextureIdList>    matTextureIds;

        SceneError LoadAlbedoTexture(const TextureI<2>*& tex,
                                     Vector3& constData,
                                     uint32_t& texCount, uint32_t& constCount,
                                     const TexturedDataNode<Vector3>& node,
                                     const TextureNodeMap& textureNodes,
                                     const std::string& scenePath);
        SceneError Load1CTexture(const TextureI<2>*& tex,
                                 float& constData,
                                 uint32_t& texCount, uint32_t& constCount,
                                 const TexturedDataNode<float>& node,
                                 const TextureNodeMap& textureNodes,
                                 const std::string& scenePath);

    protected:
    public:
        // Constructors & Destructor
                                UnrealMat(const CudaGPU& gpu)
                                    : GPUMaterialGroup<UnrealMatData, UVSurface,
                                                       UnrealDeviceFuncs>(gpu) {}
                                ~UnrealMat() = default;

        // Interface
        // Type (as string) of the primitive group
        const char*             Type() const override { return TypeName(); }
        // Allocates and Generates Data
        SceneError              InitializeGroup(const NodeListing& materialNodes,
                                                const TextureNodeMap& textures,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath) override;
        SceneError              ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath) override;
        TracerError             ConstructTextureReferences() override;

        // Material Queries
        size_t                  UsedGPUMemory() const override;
        size_t                  UsedCPUMemory() const override;
        size_t                  UsedGPUMemory(uint32_t materialId) const override;
        size_t                  UsedCPUMemory(uint32_t materialId) const override;

        uint8_t                 SampleStrategyCount() const override { return 1; };
        // No Texture
        uint8_t                 UsedTextureCount() const override;
        std::vector<uint32_t>   UsedTextureIds() const override;
};

static_assert(IsMaterialGroupClass<UnrealMat>::value,
              "UnrealMat is not a GPU Material Group Class.");