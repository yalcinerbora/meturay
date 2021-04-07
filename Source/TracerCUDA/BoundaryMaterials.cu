#include "BoundaryMaterials.cuh"
#include "CudaConstants.h"
#include "CudaConstants.hpp"
#include "TextureFunctions.h"
#include "TextureReferenceGenerators.cuh"

#include "RayLib/ColorConversion.h"
#include "RayLib/MemoryAlignment.h"

__global__
void KCRGBTextureToLuminanceArray(float* gOutLuminance,
                                  const TextureRef<2, Vector3>& gTexture,
                                  Vector2ui dimension)
{
    uint32_t totalWorkCount = dimension[0] * dimension[1];
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < totalWorkCount;
        threadId += (blockDim.x * gridDim.x))
    {
        Vector2ui id2D = Vector2ui(threadId % dimension[1],
                                   threadId / dimension[1]);
        // Convert to UV coordinates
        Vector2f uv = Vector2f(id2D) / Vector2f(dimension);
        // Bypass linear interp
        uv += Vector2f(0.5f);

        Vector3 rgb = gTexture(uv);
        gOutLuminance[threadId] = Utility::RGBToLuminance(rgb);
    }
}

SceneError BoundaryMatConstant::InitializeGroup(const NodeListing& materialNodes,
                                                const TextureNodeMap& textures,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath)
{
    constexpr const char* RADIANCE = "radiance";

    std::vector<Vector3> radianceCPU;
    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {
        std::vector<Vector3> radiances = sceneNode->AccessVector3(RADIANCE);
        radianceCPU.insert(radianceCPU.end(), radiances.begin(), radiances.end());

        // Calculate Distributions
        for(const Vector3& r : radiances)
        {
            float lum = Utility::RGBToLuminance(r);
            //luminanceDistributions.emplace_back(std::vector<float>(1, lum));
        }
        // Generate Id pairs
        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }
    }

    // Alloc etc
    size_t dRadianceSize = radianceCPU.size() * sizeof(Vector3);
    memory = std::move(DeviceMemory(dRadianceSize));
    Vector3f* dRadiances = static_cast<Vector3f*>(memory);
    CUDA_CHECK(cudaMemcpy(dRadiances, radianceCPU.data(), dRadianceSize,
                          cudaMemcpyHostToDevice));

    dData = LightMatData{dRadiances};
    return SceneError::OK;
}

SceneError BoundaryMatConstant::ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}
        
TracerError BoundaryMatConstant::LuminanceData(std::vector<float>& lumData,
                                               Vector2ui& dim,
                                               uint32_t innerId) const
{
    if(innerId >= innerIds.size())
        return TracerError::MATERIAL_CAN_NOT_GENERATE_LUMINANCE;

    Vector3 radiance;
    CUDA_CHECK(cudaMemcpy(&radiance,
               dData.dRadiances + innerId,
               sizeof(Vector3),
               cudaMemcpyDeviceToHost));
    
    lumData.push_back(Utility::RGBToLuminance(radiance));
    dim = Vector2ui(1, 1);
    return TracerError::OK;
}

SceneError BoundaryMatTextured::InitializeGroup(const NodeListing& materialNodes,
                                                const TextureNodeMap&,
                                                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                double time, const std::string& scenePath)
  
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

SceneError BoundaryMatTextured::ChangeTime(const NodeListing& materialNodes, double time,
                                           const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

TracerError BoundaryMatTextured::LuminanceData(std::vector<float>& lumData,
                                               Vector2ui& dim,
                                               uint32_t innerId) const
{   
    // TODO: Implement
    return TracerError::MATERIAL_CAN_NOT_GENERATE_LUMINANCE;
}

SceneError BoundaryMatSkySphere::InitializeGroup(const NodeListing& materialNodes,
                                                 const TextureNodeMap& textureNodes,
                                                 const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                 double time, const std::string& scenePath)
{
    constexpr const char* RADIANCE = "radiance";
    SceneError err = SceneError::OK;

    std::vector<Vector3> radianceCPU;
    uint32_t i = 0;
    for(const auto& sceneNode : materialNodes)
    {
        TextureList radianceTextures = sceneNode->AccessTextureNode(RADIANCE);
        
        // Calculate Distributions
        for(const NodeTextureStruct& texInfo : radianceTextures)
        {
            const TextureI<2, 4>* texture;
            if((err = TextureFunctions::AllocateTexture(texture,
                                                        textureMemory, texInfo,
                                                        textureNodes,
                                                        EdgeResolveType::WRAP,
                                                        InterpolationType::LINEAR,
                                                        true, true,
                                                        gpu, scenePath)) != SceneError::OK)
                return err;

            textureList.push_back(static_cast<cudaTextureObject_t>(*texture));
        }

        // Generate Id pairs
        const auto& ids = sceneNode->Ids();
        for(IdPair id : ids)
        {
            innerIds.emplace(std::make_pair(id.first, i));
            i++;
        }
    }

    size_t totalMatCount = textureList.size();
    // Alloc etc
    size_t texRefSize = totalMatCount * sizeof(Texture2DRef);
    texRefSize = Memory::AlignSize(texRefSize);

    memory = std::move(DeviceMemory(texRefSize));
    Texture2DRef* dTexReferences = static_cast<Texture2DRef*>(memory);
    dData = LightMatTexData{dTexReferences};
    return SceneError::OK;
}

SceneError BoundaryMatSkySphere::ChangeTime(const NodeListing& materialNodes, double time,
                                            const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::MATERIAL_TYPE_INTERNAL_ERROR;
}

TracerError BoundaryMatSkySphere::ConstructTextureReferences()
{
    return TracerError::OK;
}
TracerError BoundaryMatSkySphere::LuminanceData(std::vector<float>& lumData,
                                                Vector2ui& dim,
                                                uint32_t innerId) const
{
    if(innerId >= innerIds.size())
        return TracerError::MATERIAL_CAN_NOT_GENERATE_LUMINANCE;

    // Size
    dim = textureList[innerId].Dim();
    uint32_t totalCount = dim[0] * dim[1];

    // Allocate Temp GPU Memory for Lum values
    // then call a kernel to populate luminances
    DeviceMemory lumValueMem(totalCount * sizeof(float));
    float* dLumArray = static_cast<float*>(lumValueMem);

    // Use your own gpu since texture resides there
    gpu.GridStrideKC_X(0, (cudaStream_t)0, totalCount,
                       // Kernel
                       KCRGBTextureToLuminanceArray,
                       // Args
                       dLumArray,
                       dData.dRadianceTextures[innerId],
                       dim);

    lumData.resize(totalCount);
    CUDA_CHECK(cudaMemcpy(lumData.data(),
               dLumArray, totalCount * sizeof(float),
               cudaMemcpyDeviceToHost));
    return TracerError::OK;
}