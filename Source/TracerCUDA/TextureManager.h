#pragma once
/**

    Texture Memory Manager for a single GPU
    
    Load textures to GPU memory responsible for
    creation and deletion of textures

*/

#include <vector>
#include <map>

//#include "CudaConstants.h"
#include "Texture.cuh"
#include "SamplerI.cuh"

#include "RayLib/SceneNodeI.h"

class CudaGPU;

enum class SamplerLayout1ChannelType
{
    // 1 Channel Requests
    R, G, B, A,
};
enum class SamplerLayout2ChannelType
{
    // 2 Channel Requests (Swizzle skipped)
    RG, RB, RA, GB, GA, BA,
};
enum class SamplerLayout3ChannelType
{
    // 3 Channel Requests (Swizzle skipped)
    RGB, RGA, RBA, GBA,
};
enum class SamplerLayout4ChannelType
{
    // 4 Channel Requests (Swizzle skipped)
    RGBA
};

class TextureManager
{
    private:
        const CudaGPU& gpu;

        // 2D Textures
        std::map<uint32_t, Texture2D<float>>       texture2D1C;
        std::map<uint32_t, Texture2D<Vector2>>     texture2D2C;
        std::map<uint32_t, Texture2D<Vector4>>     texture2D4C;

        // 2D 3 Channel Textures
        std::map<uint32_t, Texture2D<float>>       texture2D3CR;
        std::map<uint32_t, Texture2D<float>>       texture2D3CG;
        std::map<uint32_t, Texture2D<float>>       texture2D3CB;
        
        // Samplers
        //SamplerI<2, 3, float>*

    protected:
    public:
        // Constructors & Destructor
                                        TextureManager(const CudaGPU&);
                                        TextureManager(const TextureManager&) = delete;
                                        TextureManager(TextureManager&&) = default;
        TextureManager&                 operator=(const TextureManager&) = delete;
        TextureManager&                 operator=(TextureManager&&) = default;
                                        ~TextureManager() = default;

        // Methods
        SceneError                      Initialize(const SceneNodePtr& textureList);

        // 1D Texture Sampler Acquisiton

        // 2D Texture Sampler Acquisiton
        const SamplerI<2, 1, float>*    RequestSampler(uint32_t texId, 
                                                       SamplerLayout1ChannelType);
        const SamplerI<2, 2, float>*    RequestSampler(uint32_t texId,
                                                       SamplerLayout2ChannelType);
        const SamplerI<2, 3, float>*    RequestSampler(uint32_t texId,
                                                       SamplerLayout3ChannelType);
        const SamplerI<2, 4, float>*    RequestSampler(uint32_t texId,
                                                       SamplerLayout4ChannelType);

        // 3D Texture Sampler Acquisiton

        // Accessors
        const CudaGPU&                  GPU() const;
        size_t                          UsedGPUMemory() const;
        size_t                          UsedCPUMemory() const;
        size_t                          UsedGPUMemory(uint32_t textureId) const;
        size_t                          UsedCPUMemory(uint32_t texturelId) const;
};