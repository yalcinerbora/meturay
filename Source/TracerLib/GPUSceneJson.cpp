#include "GPUSceneJson.h"

#include "RayLib/SceneIO.h"
#include "RayLib/Types.h"
#include "RayLib/Log.h"
#include "RayLib/SceneNodeI.h"

#include "TracerLogicI.h"
#include "GPUAcceleratorI.h"
#include "GPUPrimitiveI.h"
#include "GPUMaterialI.h"
#include "TracerLogicGeneratorI.h"
#include "ScenePartitionerI.h"
#include "SceneNodeJson.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <set>

SceneError GPUSceneJson::OpenFile(const std::u8string& fileName)
{
    // TODO: get a lightweight lexer and strip comments
    // from json since json does not support comments
    // now its only pure json iterating over a scene is
    // not convenient without comments.

    // Always assume filenames are UTF-8
    const auto path = std::filesystem::path(fileName);
    std::ifstream file(path);

    if(!file.is_open()) return SceneError::FILE_NOT_FOUND;
    // Parse Json
    sceneJson = std::make_unique<nlohmann::json>();
    file >> (*sceneJson);
    return SceneError::OK;
}

GPUSceneJson::GPUSceneJson(const std::u8string& fileName,
                           ScenePartitionerI& partitioner,
                           TracerLogicGeneratorI& lg,
                           const SurfaceLoaderGeneratorI& sl)
    : logicGenerator(lg)
    , partitioner(partitioner)
    , surfaceLoaderGenerator(sl)
    , maxAccelIds(Vector2i(-1))
    , maxMatIds(Vector2i(-1))
    , baseBoundaryMatKey(HitKey::InvalidKey)
    , fileName(fileName)
    , currentTime(0.0)
    , dLights(nullptr)
    , dTransforms(nullptr)
    , sceneJson(nullptr)
{}

bool GPUSceneJson::FindNode(const nlohmann::json*& jsn, const char* c)
{
    const auto i = sceneJson->find(c);
    bool found = (i != sceneJson->end());
    if(found) jsn = &(*i);
    return found;
};

SceneError GPUSceneJson::GenIdLookup(IndexLookup& result,
                                     const nlohmann::json& array, 
                                     IdBasedNodeType t)
{
    static constexpr uint32_t MAX_UINT32 = std::numeric_limits<uint32_t>::max();

    result.clear();
    uint32_t i = 0;
    for(const auto& jsn : array)
    {
        const nlohmann::json& ids = jsn[SceneIO::ID];
        if(!ids.is_array())
        {
            auto r = result.emplace(jsn[SceneIO::ID], std::make_pair(i, MAX_UINT32));
            if(!r.second)
            {
                unsigned int i = static_cast<int>(SceneError::DUPLICATE_MATERIAL_ID) + t;
                return static_cast<SceneError::Type>(i);
            }
        }
        else
        {
            uint32_t j = 0;
            for(const auto& id : ids)
            {
                auto r = result.emplace(id, std::make_pair(i, j));
                if(!r.second)
                {
                    unsigned int i = static_cast<int>(SceneError::DUPLICATE_MATERIAL_ID) + t;
                    return static_cast<SceneError::Type>(i);
                }
                j++;
            }
        }
        i++;
    }
    return SceneError::OK;
}

SceneError GPUSceneJson::GenerateConstructionData(// Striped Listings (Striped from unsued nodes)
                                                  PrimitiveNodeList& primGroupNodes,
                                                  //
                                                  MaterialNodeList& matGroupNodes,
                                                  MaterialBatchList& matBatchListings,
                                                  AcceleratorBatchList& requiredAccelListings,
                                                  // Base Accelerator required data
                                                  std::map<uint32_t, uint32_t>& surfaceTransformIds,
                                                  double time)
{
    const nlohmann::json* surfaces = nullptr;
    const nlohmann::json* primitives = nullptr;
    const nlohmann::json* materials = nullptr;
    const nlohmann::json* lights = nullptr;
    IndexLookup primList;
    IndexLookup materialList;

    // Lambdas for cleaner code
    auto AttachMatBatch = [&](const std::string& primType,
                              const std::string& matType,
                              const NodeId matId)
    {
        // Generate its mat batch also
        MatBatchData batchData = MatBatchData
        {
            primType,
            matType,
            std::set<NodeId>()
        };
        const auto& matBatch = matBatchListings.emplace(primType + matType,
                                                        batchData).first;
        matBatch->second.matIds.emplace(matId);
    };
    auto AttachMatAll = [&] (const std::string& primType,
                             const NodeId matId)
    {
        if(auto loc = materialList.find(matId); loc != materialList.end())
        {
            const NodeIndex nIndex = loc->second.first;
            const InnerIndex iIndex = loc->second.second;
            const auto& jsnNode = (*materials)[nIndex];

            const std::string matGroupType = jsnNode[SceneIO::TYPE];
            auto& matSet = matGroupNodes.emplace(matGroupType, NodeListing()).first->second;
            auto& node = *matSet.emplace(std::make_unique<SceneNodeJson>(jsnNode, nIndex)).first;
            node->AddIdIndexPair(matId, iIndex);
            AttachMatBatch(primType, matGroupType, matId);
        }
        else return SceneError::MATERIAL_ID_NOT_FOUND;
        return SceneError::OK;
    };

    // Function Start
    SceneError e = SceneError::OK;

    // Load Id Based Arrays
    if(!FindNode(surfaces, SceneIO::SURFACE_BASE)) return SceneError::SURFACES_ARRAY_NOT_FOUND;
    if(!FindNode(primitives, SceneIO::PRIMITIVE_BASE)) return SceneError::PRIMITIVES_ARRAY_NOT_FOUND;
    if(!FindNode(materials, SceneIO::MATERIAL_BASE)) return SceneError::MATERIALS_ARRAY_NOT_FOUND;
    if(!FindNode(lights, SceneIO::LIGHT_BASE)) return SceneError::LIGHTS_ARRAY_NOT_FOUND;
    if((e = GenIdLookup(primList, *primitives, PRIMITIVE)) != SceneError::OK) 
        return e;
    if((e = GenIdLookup(materialList, *materials, MATERIAL)) != SceneError::OK)
        return e;

    // Iterate over surfaces
    // and collect data for groups and batches
    uint32_t surfId = 0;
    for(const auto& jsn : (*surfaces))
    {
        SurfaceStruct surf = SceneIO::LoadSurface(jsn, time);
        // Start loading mats and surface datas
        // Iterate mat surface pairs
        std::string primGroupType;
        for(int i = 0; i < surf.pairCount; i++)
        {
            const auto& pairs = surf.matPrimPairs;
            const uint32_t primId = pairs[i].second;
            const uint32_t matId = pairs[i].first;

            // Check if primitive exists
            // add it to primitive group list for later construction
            if(auto loc = primList.find(primId); loc != primList.end())
            {
                const NodeIndex nIndex = loc->second.first;
                const InnerIndex iIndex = loc->second.second;
                const auto& jsnNode = (*primitives)[nIndex];

                std::string currentType = jsnNode[SceneIO::TYPE];
                if((i != 0) && primGroupType != currentType)
                    return SceneError::PRIM_TYPE_NOT_CONSISTENT_ON_SURFACE;
                else primGroupType = currentType;
                auto& primSet = primGroupNodes.emplace(primGroupType, NodeListing()).first->second;
                auto& node = *primSet.emplace(std::make_unique<SceneNodeJson>(jsnNode, nIndex)).first;
                node->AddIdIndexPair(primId, iIndex);
            }
            else return SceneError::PRIMITIVE_ID_NOT_FOUND;

            // Do Material attachments
            if((e = AttachMatAll(primGroupType, matId)) != SceneError::OK)
                return e;
        }
        // Generate Accelerator Group
        const std::string acceleratorGroupType = surf.acceleratorType + primGroupType;
        AccelGroupData accGData =
        {
            acceleratorGroupType,
            primGroupType,
            std::map<uint32_t, IdPairs>()
        };
        const auto& result = requiredAccelListings.emplace(acceleratorGroupType, accGData).first;
        result->second.matPrimIdPairs.emplace(surfId, surf.matPrimPairs);

        // Generate transform pair also
        surfaceTransformIds.emplace(surfId, surf.transformId);
        surfId++;
    }
    // Additionally For Lights and Base Boundary Material 
    // Generate a Empty primitive (if not already requrested)
    primGroupNodes.emplace(BaseConstants::EMPTY_PRIMITIVE_NAME, NodeListing());
    // Generate Material listing and material group
    // For Lights
    for(const auto& jsn : (*lights))
    {
        LightStruct l = SceneIO::LoadLight(jsn, time);
        if((e = AttachMatAll(BaseConstants::EMPTY_PRIMITIVE_NAME, l.matId)) != SceneError::OK)
           return e;
    }
    // Finally Boundary Material
    const nlohmann::json* baseMatNode = nullptr;
    if(!FindNode(baseMatNode, SceneIO::BASE_OUTSIDE_MATERIAL))
        return SceneError::BASE_BOUND_MAT_NODE_NOT_FOUND;
    if((e = AttachMatAll(BaseConstants::EMPTY_PRIMITIVE_NAME,
                         SceneIO::LoadNumber<uint32_t>(*baseMatNode, time))) != SceneError::OK)
        return e;
    return e;
}

SceneError GPUSceneJson::GenerateMaterialGroups(const MultiGPUMatNodes& matGroupNodes,
                                                double time)
{
    // Generate Partitioned Material Groups
    SceneError e = SceneError::OK;
    for(const auto& matGroup : matGroupNodes)
    {
        const std::string& matTypeName = matGroup.first.first;
        const int gpuId = matGroup.first.second;
        const auto& matNodes = matGroup.second;
        //
        GPUMaterialGroupI* matGroup = nullptr;
        if(e = logicGenerator.GenerateMaterialGroup(matGroup, matTypeName, gpuId))
            return e;
        if(e = matGroup->InitializeGroup(matNodes, time))
            return e;
    }
    return e;
}

SceneError GPUSceneJson::GenerateMaterialBatches(MaterialKeyListing& allMatKeys,
                                                 const MultiGPUMatBatches& materialBatches,
                                                 double time)
{
    SceneError e = SceneError::OK;
    // First do materials
    uint32_t batchId = NullBatchId;
    for(const auto& requiredMat : materialBatches)
    {
        batchId++;
        if(batchId >= (1 << HitKey::BatchBits))
            return SceneError::TOO_MANY_MATERIAL_GROUPS;

        const int gpuId = requiredMat.first.second;
        const std::string& batchName = requiredMat.first.first;
        const std::string& matTName = requiredMat.second.matType;
        const std::string& primTName = requiredMat.second.primType;

        GPUPrimitiveGroupI* pGroup = nullptr;
        GPUMaterialGroupI* mGroup = nullptr;
        if((e = logicGenerator.GeneratePrimitiveGroup(pGroup, primTName)) != SceneError::OK)
            return e;
        if((e = logicGenerator.GenerateMaterialGroup(mGroup, matTName, gpuId)) != SceneError::OK)
            return e;

        // Generation
        GPUMaterialBatchI* matBatch = nullptr;
        if((e = logicGenerator.GenerateMaterialBatch(matBatch,
                                                     *mGroup,
                                                     *pGroup,
                                                     batchId)) != SceneError::OK)
            return e;

        // Generate Keys
        // Find inner ids of those materials
        // And combine a key
        const GPUMaterialGroupI& matGroup = *mGroup;
        for(const auto& matId : requiredMat.second.matIds)
        {
            uint32_t innerId = matGroup.InnerId(matId);
            HitKey key = HitKey::CombinedKey(batchId, innerId);
            allMatKeys.emplace(std::make_pair(primTName, matId), key);

            maxMatIds = Vector2i::Max(maxMatIds, Vector2i(batchId, innerId));
        }
    }
    return e;
}

SceneError GPUSceneJson::GeneratePrimitiveGroups(const PrimitiveNodeList& primGroupNodes,
                                                 double time)
{
    // Generate Primitive Groups
    SceneError e = SceneError::OK;
    for(const auto& primGroup : primGroupNodes)
    {
        std::string primTypeName = primGroup.first;
        const auto& primNodes = primGroup.second;
        //
        GPUPrimitiveGroupI* primGroup = nullptr;
        if(e = logicGenerator.GeneratePrimitiveGroup(primGroup, primTypeName))
            return e;
        if(e = primGroup->InitializeGroup(primNodes, time, surfaceLoaderGenerator))
            return e;
    }
    return e;
}

SceneError GPUSceneJson::GenerateAccelerators(std::map<uint32_t, AABB3>& accAABBs,
                                              std::map<uint32_t, HitKey>& accHitKeyList,
                                              //
                                              const AcceleratorBatchList& acceleratorBatchList,
                                              const MaterialKeyListing& matHitKeyList,
                                              double time)
{
    SceneError e = SceneError::OK;
    uint32_t accelBatch = NullBatchId;
    // Accelerator Groups & Batches and surface hit keys
    for(const auto& accelGroupBatch : acceleratorBatchList)
    {
        // Too many accelerators
        accelBatch++;
        if(accelBatch >= (1 << HitKey::BatchBits))
            return SceneError::TOO_MANY_ACCELERATOR_GROUPS;

        const uint32_t accelId = accelBatch;
        const std::string& accelGroupName = accelGroupBatch.second.accelType;
        const auto& primTName = accelGroupBatch.second.primType;
        const auto& pairsList = accelGroupBatch.second.matPrimIdPairs;

        // Fetch Primitive
        GPUPrimitiveGroupI* pGroup = nullptr;
        if((e = logicGenerator.GeneratePrimitiveGroup(pGroup, primTName)) != SceneError::OK)
            return e;

        // Group Generation
        GPUAcceleratorGroupI* aGroup = nullptr;
        if((e = logicGenerator.GenerateAcceleratorGroup(aGroup, *pGroup, dTransforms, accelGroupName)) != SceneError::OK)
            return e;
        if((e = aGroup->InitializeGroup(matHitKeyList, pairsList, time)) != SceneError::OK)
            return e;

        // Batch Generation
        GPUAcceleratorBatchI* aBatch = nullptr;
        if((e = logicGenerator.GenerateAcceleratorBatch(aBatch, *aGroup, *pGroup,
                                                        accelId)) != SceneError::OK)
            return e;

        // Now Keys
        // Generate Accelerator Keys...
        const GPUAcceleratorGroupI& accGroup = *aGroup;
        for(const auto& pairings : accelGroupBatch.second.matPrimIdPairs)
        {
            const uint32_t surfId = pairings.first;
            uint32_t innerId = accGroup.InnerId(surfId);
            HitKey key = HitKey::CombinedKey(accelId, innerId);
            accHitKeyList.emplace(surfId, key);

            maxAccelIds = Vector2i::Max(maxAccelIds, Vector2i(accelId, innerId));

            // Attach keys of accelerators
            accHitKeyList.emplace(surfId, key);
        }

        // Find AABBs of these surfaces
        // For base Accelerator generation
        std::map<uint32_t, AABB3> aabbs;
        for(const auto& pairs : pairsList)
        {
            AABB3 combinedAABB = ZeroAABB3;
            const IdPairs& pList = pairs.second;
            // Merge aabbs of the surfaces
            for(const auto& p : pList)
            {
                if(p.first == std::numeric_limits<uint32_t>::max()) break;

                AABB3 aabb = accGroup.PrimitiveGroup().PrimitiveBatchAABB(p.second);
                combinedAABB = combinedAABB.Union(aabb);
            }
            accAABBs.emplace(pairs.first, std::move(combinedAABB));
        }
    }
    return e;
}

SceneError GPUSceneJson::GenerateBaseAccelerator(const std::map<uint32_t, AABB3>& accAABBs,
                                                 const std::map<uint32_t, HitKey>& accHitKeyList,
                                                 const std::map<uint32_t, uint32_t>& surfaceTransformIds,
                                                 double time)
{
    SceneError e = SceneError::OK;
    // Generate Surface Listings
    std::map<uint32_t, BaseLeaf> surfaceListings;
    for(const auto& pairs : surfaceTransformIds)
    {
        const uint32_t id = pairs.first;
        const AABB3f& aabb = accAABBs.at(id);
        const HitKey& key = accHitKeyList.at(id);

        BaseLeaf leaf =
        {
            aabb.Min(),
            key,
            aabb.Max(),
            pairs.second
        };
        surfaceListings.emplace(pairs.first, leaf);
    }

    // Find Base Accelerator Type and generate
    const nlohmann::json* baseAccel = nullptr;
    if(!FindNode(baseAccel, SceneIO::BASE_ACCELERATOR_BASE))
        return SceneError::BASE_ACCELERATOR_NODE_NOT_FOUND;
    const std::string baseAccelType = (*baseAccel);

    // Generate Base Accelerator..
    GPUBaseAcceleratorI* baseAccelerator = nullptr;
    if((e = logicGenerator.GenerateBaseAccelerator(baseAccelerator, baseAccelType)) != SceneError::OK)
        return e;
    if((e = baseAccelerator->Initialize(surfaceListings)) != SceneError::OK)
        return e;
    return e;
}

SceneError GPUSceneJson::FindBoundaryMaterial(const MaterialKeyListing& matHitKeyList,
                                              double time)
{
    SceneError e = SceneError::OK;
    NodeListing nodeList;

    const nlohmann::json* node = nullptr;
    if(!FindNode(node, SceneIO::BASE_OUTSIDE_MATERIAL))
        return SceneError::BASE_BOUND_MAT_NODE_NOT_FOUND;

    // From that node find equavilent material,
    auto primIdPair = std::make_pair(std::string(BaseConstants::EMPTY_PRIMITIVE_NAME),
                                     SceneIO::LoadNumber<uint32_t>(*node, time));
    auto loc = matHitKeyList.find(primIdPair);
    if(loc == matHitKeyList.end())
        return SceneError::MATERIAL_ID_NOT_FOUND;

    baseBoundaryMatKey = loc->second;
    return e;
}

SceneError GPUSceneJson::LoadCommon(double time)
{
    SceneError e = SceneError::OK;

    // CPU Temp Data
    int i;
    std::vector<LightStruct> lightsCPU;
    std::vector<TransformStruct> transformsCPU;

    // Lights
    const nlohmann::json* lightsJson = nullptr;
    if(!FindNode(lightsJson, SceneIO::LIGHT_BASE))
        return SceneError::LIGHTS_ARRAY_NOT_FOUND;
    i = 0;
    lightsCPU.resize(lightsJson->size());
    for(const auto& lightJson : (*lightsJson))
    {
        lightsCPU[i] = SceneIO::LoadLight(lightJson, time);
        i++;
    }
    
    // Transforms
    const nlohmann::json* transformsJson = nullptr;
    if(!FindNode(transformsJson, SceneIO::TRANSFORM_BASE))
        return SceneError::TRANSFORMS_ARRAY_NOT_FOUND;
    i = 0;
    transformsCPU.resize(transformsJson->size());
    for(const auto& transformJson : (*transformsJson))
    {
        transformsCPU[i] = SceneIO::LoadTransform(transformJson, time);
        i++;
    }

    // Allocate GPU and Load
    size_t transformSize = transformsCPU.size() * sizeof(TransformStruct);
    transformSize = AlignByteCount * ((transformSize + (AlignByteCount - 1)) / AlignByteCount);
    size_t lightSize = lightsCPU.size() * sizeof(LightStruct);

    memory = DeviceMemory(transformSize + lightSize);
    if(transformsCPU.size() != 0)
    {
        dTransforms = reinterpret_cast<TransformStruct*>(static_cast<Byte*>(memory));
        CUDA_CHECK(cudaMemcpy(dTransforms, transformsCPU.data(),
                              transformsCPU.size() * sizeof(TransformStruct),
                              cudaMemcpyHostToDevice));
    }
    if(lightsCPU.size() != 0)
    {
        dLights = reinterpret_cast<LightStruct*>(static_cast<Byte*>(memory) + transformSize);
        CUDA_CHECK(cudaMemcpy(dLights, lightsCPU.data(), lightsCPU.size() * sizeof(LightStruct),
                              cudaMemcpyHostToDevice));
    }

    // Now Load Camera
    const nlohmann::json* camerasJson = nullptr;
    if(!FindNode(camerasJson, SceneIO::CAMERA_BASE))
        return SceneError::CAMERAS_ARRAY_NOT_FOUND;
    i = 0;
    cameraMemory.resize(camerasJson->size());
    for(const auto& cameraJson : (*camerasJson))
    {
        cameraMemory[i] = SceneIO::LoadCamera(cameraJson, time);
        i++;
    }
    return e;
}

SceneError GPUSceneJson::LoadLogicRelated(double time)
{
    SceneError e = SceneError::OK;
    // Group Data
    PrimitiveNodeList primGroupNodes;
    //
    MaterialNodeList matGroupNodes;
    MaterialBatchList matListings;
    AcceleratorBatchList accelListings;
    std::map<uint32_t, uint32_t> surfaceTransformIds;

    // Parse Json and find necessary nodes
    if((e = GenerateConstructionData(primGroupNodes,
                                     matGroupNodes,
                                     matListings,
                                     accelListings,
                                     surfaceTransformIds,
                                     time)) != SceneError::OK)
        return e;

    // Partition Material Data to Multi GPU Material Data
    int boundaryMaterialGPUId;
    MultiGPUMatNodes multiGPUMatNodes;
    MultiGPUMatBatches multiGPUMatBatches;
    if((e = partitioner.PartitionMaterials(multiGPUMatNodes,
                                           multiGPUMatBatches,
                                           boundaryMaterialGPUId,
                                           //
                                           matGroupNodes,
                                           matListings)))
        return e;

    // Using those constructs generate
    // Primitive Groups
    if((e = GeneratePrimitiveGroups(primGroupNodes, time)) != SceneError::OK)
        return e;

    // Material Groups
    if((e = GenerateMaterialGroups(multiGPUMatNodes, time)) != SceneError::OK)
        return e;

    // Material Batches
    MaterialKeyListing allMaterialKeys;
    if((e = GenerateMaterialBatches(allMaterialKeys,
                                    multiGPUMatBatches,
                                    time)) != SceneError::OK)
        return e;

    // Accelerators
    std::map<uint32_t, AABB3> accAABBs;
    std::map<uint32_t, HitKey> accHitKeyList;
    if((e = GenerateAccelerators(accAABBs, accHitKeyList, accelListings,
                                 allMaterialKeys, time)) != SceneError::OK)
        return e;

    // Base Accelerator
    if((e = GenerateBaseAccelerator(accAABBs, accHitKeyList,
                                    surfaceTransformIds, time)) != SceneError::OK)
        return e;

    // Finally Boundary Material
    if((e = FindBoundaryMaterial(allMaterialKeys, time)) != SceneError::OK)
       return e;

    // MaxIds are generated but those are inclusive
    // Make them exclusve
    maxAccelIds += Vector2i(1);
    maxMatIds += Vector2i(1);

    // Everything is generated!

    // All of the data is generated
    return SceneError::OK;
}

SceneError GPUSceneJson::ChangeCommon(double time)
{
    // TODO:
    return SceneError::OK;
}

SceneError GPUSceneJson::ChangeLogicRelated(double time)
{
    // TODO:
    return SceneError::BASE_ACCELERATOR_NODE_NOT_FOUND;
}

size_t GPUSceneJson::UsedGPUMemory()
{
    //return transformMemory.Size() + lightMemory.Size();
    return 0;
}

size_t GPUSceneJson::UsedCPUMemory()
{
    //return cameraMemory.size() * sizeof(CameraPerspective);
    return 0;
}

SceneError GPUSceneJson::LoadScene(double time)
{
    SceneError e = SceneError::OK;
    try
    {
        if((e = OpenFile(fileName)) != SceneError::OK)
           return e;
        if((e = LoadCommon(time)) != SceneError::OK)
           return e;
        e = LoadLogicRelated(time);
    }
    catch (SceneException const& e)
    {
        return e;
    }
    catch(nlohmann::json::parse_error const& e)
    {
        METU_ERROR_LOG("Error: %s", e.what());
        return SceneError::JSON_FILE_PARSE_ERROR;
    }
    return e;
}

SceneError GPUSceneJson::ChangeTime(double time)
{
    SceneError e = SceneError::OK;
    try
    {
        if((e = OpenFile(fileName)) != SceneError::OK)
            return e;
        if((e = ChangeCommon(time)) != SceneError::OK)
            return e;
        e = ChangeLogicRelated(time);
    }
    catch(SceneException const& e)
    {
        return e;
    }
    catch(std::exception const&)
    {
        return SceneError::JSON_FILE_PARSE_ERROR;
    }
    return e;
}

Vector2i GPUSceneJson::MaxMatIds()
{
    return maxMatIds;
}

Vector2i GPUSceneJson::MaxAccelIds()
{
    return maxAccelIds;
}

HitKey GPUSceneJson::BaseBoundaryMaterial()
{
    return baseBoundaryMatKey;
}

const LightStruct* GPUSceneJson::LightsGPU() const
{
    return dLights;
}

const TransformStruct* GPUSceneJson::TransformsGPU() const
{
    return dTransforms;
}

const CameraPerspective* GPUSceneJson::CamerasCPU() const
{
    return cameraMemory.data();
}