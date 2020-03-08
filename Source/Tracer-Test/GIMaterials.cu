//#include "GIMaterials.cuh"
//#include "MaterialNodeReaders.h"
//
//SceneError BasicPathTraceMat::InitializeGroup(const NodeListing& materialNodes, double time,
//                                              const std::string& scenePath)
//{
//    constexpr const char* ALBEDO = "albedo";
//
//    std::vector<Vector3> albedoCPU;
//    uint32_t i = 0;
//    for(const auto& sceneNode : materialNodes)
//    {
//        std::vector<Vector3> albedos = sceneNode->AccessVector3(ALBEDO);
//        albedoCPU.insert(albedoCPU.end(), albedos.begin(), albedos.end());
//
//        const auto& ids = sceneNode->Ids();
//        for(IdPair id : ids)
//        {
//            innerIds.emplace(std::make_pair(id.first, i));
//            i++;
//        }
//    }
//
//    // Alloc etc
//    size_t dAlbedoSize = albedoCPU.size() * sizeof(Vector3);
//    memory = std::move(DeviceMemory(dAlbedoSize));
//    Vector3f* dAlbedo = static_cast<Vector3f*>(memory);
//    CUDA_CHECK(cudaMemcpy(dAlbedo, albedoCPU.data(), dAlbedoSize,
//                          cudaMemcpyHostToDevice));
//
//    dData = AlbedoMatData{dAlbedo};
//    return SceneError::OK;
//}
//
//SceneError BasicPathTraceMat::ChangeTime(const NodeListing& materialNodes, double time,
//                                         const std::string& scenePath)
//{
//    // TODO: Implement
//    return SceneError::NO_LOGIC_FOR_MATERIAL;
//}
//
//SceneError LightBoundaryMat::InitializeGroup(const NodeListing& materialNodes, double time,
//                                             const std::string& scenePath)
//{
//    constexpr const char* IRRADIANCE = "radiance";
//
//    std::vector<Vector3> irradianceCPU;
//    uint32_t i = 0;
//    for(const auto& sceneNode : materialNodes)
//    {
//        std::vector<Vector3> irradiances = sceneNode->AccessVector3(IRRADIANCE);
//        irradianceCPU.insert(irradianceCPU.end(), irradiances.begin(), irradiances.end());
//
//        const auto& ids = sceneNode->Ids();
//        for(IdPair id : ids)
//        {
//            innerIds.emplace(std::make_pair(id.first, i));
//            i++;
//        }
//    }
//
//    // Alloc etc
//    size_t dIrradianceSize = irradianceCPU.size() * sizeof(Vector3);
//    memory = std::move(DeviceMemory(dIrradianceSize));
//    Vector3f* dIrradiance = static_cast<Vector3f*>(memory);
//    CUDA_CHECK(cudaMemcpy(dIrradiance, irradianceCPU.data(), dIrradianceSize,
//               cudaMemcpyHostToDevice));
//
//    dData = IrradianceMatData{dIrradiance};
//    return SceneError::OK;
//}
//
//SceneError LightBoundaryMat::ChangeTime(const NodeListing& materialNodes, double time,
//                                        const std::string& scenePath)
//{
//    // TODO: Implement
//    return SceneError::NO_LOGIC_FOR_MATERIAL;
//}
//
//SceneError BasicReflectPTMat::InitializeGroup(const NodeListing& materialNodes, double time,
//                                              const std::string& scenePath)
//{
//    constexpr const char* ALBEDO = "albedo";
//    constexpr const char* ROUGHNESS = "roughness";
//
//    std::vector<Vector4> matDataCPU;
//    uint32_t i = 0;
//    for(const auto& sceneNode : materialNodes)
//    {
//        std::vector<Vector3> albedos = sceneNode->AccessVector3(ALBEDO);
//        std::vector<float> rougnessList = sceneNode->AccessFloat(ROUGHNESS);
//
//        const auto& ids = sceneNode->Ids();
//        for(IdPair id : ids)
//        {
//            Vector4 data = Vector4(albedos[i], rougnessList[i]);
//            matDataCPU.push_back(data);
//
//            innerIds.emplace(std::make_pair(id.first, i));
//            i++;
//        }
//    }
//
//    // Alloc etc
//    size_t dMatDataSize = matDataCPU.size() * sizeof(Vector4);
//    memory = std::move(DeviceMemory(dMatDataSize));
//    Vector4f* dMemory = static_cast<Vector4f*>(memory);
//    CUDA_CHECK(cudaMemcpy(dMemory, matDataCPU.data(), dMatDataSize,
//                          cudaMemcpyHostToDevice));
//
//    dData = ReflectMatData{dMemory};
//    return SceneError::OK;
//}
//
//SceneError BasicReflectPTMat::ChangeTime(const NodeListing& materialNodes, double time,
//                                         const std::string& scenePath)
//{
//    // TODO: Implement
//    return SceneError::NO_LOGIC_FOR_MATERIAL;
//}
//
//
//SceneError BasicRefractPTMat::InitializeGroup(const NodeListing& materialNodes, double time,
//                                              const std::string& scenePath)
//{
//    constexpr const char* ALBEDO = "albedo";
//    constexpr const char* INDEX = "refractionIndex";
//
//    std::vector<Vector4> matDataCPU;
//    uint32_t i = 0;
//    for(const auto& sceneNode : materialNodes)
//    {
//        std::vector<Vector3> albedos = sceneNode->AccessVector3(ALBEDO);
//        std::vector<float> indices = sceneNode->AccessFloat(INDEX);
//       
//        const auto& ids = sceneNode->Ids();
//        for(IdPair id : ids)
//        {
//            Vector4 data = Vector4(albedos[i], indices[i]);
//            matDataCPU.push_back(data);
//            
//            innerIds.emplace(std::make_pair(id.first, i));
//            i++;
//        }
//    }
//
//    // Alloc etc
//    size_t dMatDataSize = matDataCPU.size() * sizeof(Vector4);
//    memory = std::move(DeviceMemory(dMatDataSize));
//    Vector4f* dMemory = static_cast<Vector4f*>(memory);
//    CUDA_CHECK(cudaMemcpy(dMemory, matDataCPU.data(), dMatDataSize,
//               cudaMemcpyHostToDevice));
//
//    dData = RefractMatData{dMemory};
//    return SceneError::OK;
//}
//
//SceneError BasicRefractPTMat::ChangeTime(const NodeListing& materialNodes, double time,
//                                        const std::string& scenePath)
//{
//    // TODO: Implement
//    return SceneError::NO_LOGIC_FOR_MATERIAL;
//}
//
//// Diffuse
//DEFINE_MATH_BATCH(BasicPTTriangleBatch, TracerBasic, GPUEventEstimatorBasic,
//                  BasicPathTraceMat, GPUPrimitiveTriangle, BasicSurfaceFromTri);
//
//DEFINE_MATH_BATCH(BasicPTSphereBatch, TracerBasic, GPUEventEstimatorBasic,
//                  BasicPathTraceMat, GPUPrimitiveSphere, BasicSurfaceFromSphr);
//
//// Light
//DEFINE_MATH_BATCH(LightBoundaryBatch, TracerBasic, GPUEventEstimatorBasic,
//                  LightBoundaryMat, GPUPrimitiveEmpty, EmptySurfaceFromEmpty);
//
//DEFINE_MATH_BATCH(LightBoundaryTriBatch, TracerBasic, GPUEventEstimatorBasic,
//                  LightBoundaryMat, GPUPrimitiveTriangle, EmptySurfaceFromTri);
//
//DEFINE_MATH_BATCH(LightBoundarySphrBatch, TracerBasic, GPUEventEstimatorBasic,
//                  LightBoundaryMat, GPUPrimitiveSphere, EmptySurfaceFromSphr);
//
//// Reflect
//DEFINE_MATH_BATCH(ReflectPTTriangleBatch, TracerBasic, GPUEventEstimatorBasic,
//                  BasicReflectPTMat, GPUPrimitiveTriangle, BasicSurfaceFromTri);
//
//DEFINE_MATH_BATCH(ReflectPTSphereBatch, TracerBasic, GPUEventEstimatorBasic,
//                  BasicReflectPTMat, GPUPrimitiveSphere, BasicSurfaceFromSphr);
//
//// Refract
//DEFINE_MATH_BATCH(RefractPTTriangleBatch, TracerBasic, GPUEventEstimatorBasic,
//                  BasicRefractPTMat, GPUPrimitiveTriangle, BasicSurfaceFromTri);
//
//DEFINE_MATH_BATCH(RefractPTSphereBatch, TracerBasic, GPUEventEstimatorBasic,
//                  BasicRefractPTMat, GPUPrimitiveSphere, BasicSurfaceFromSphr);