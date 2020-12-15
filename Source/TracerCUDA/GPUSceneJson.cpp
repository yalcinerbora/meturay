#include "GPUSceneJson.h"

#include "RayLib/SceneIO.h"
#include "RayLib/Types.h"
#include "RayLib/Log.h"
#include "RayLib/SceneNodeI.h"
#include "RayLib/SceneNodeNames.h"
#include "RayLib/StripComments.h"

#include "GPUCameraI.h"
#include "GPULightI.h"
#include "GPUAcceleratorI.h"
#include "GPUPrimitiveI.h"
#include "GPUMaterialI.h"
#include "GPUMediumI.h"

#include "TracerLogicGeneratorI.h"
#include "ScenePartitionerI.h"
#include "MangledNames.h"

#include "SceneNodeJson.h"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <set>
#include <regex>
#include <execution>

void GPUSceneJson::ExpandHitStructSize(const GPUPrimitiveGroupI& pg)
{
    uint32_t newHit = pg.PrimitiveHitSize();
    // Properly Align
    newHit = ((newHit + sizeof(uint32_t) - 1) / sizeof(uint32_t)) * sizeof(uint32_t);
    hitStructSize = std::max(hitStructSize, newHit);
}

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
    auto stream = Utility::StripComments(file);

    // Parse Json
    sceneJson = std::make_unique<nlohmann::json>();
    stream >> (*sceneJson);
    return SceneError::OK;
}

GPUSceneJson::GPUSceneJson(const std::u8string& fileName,
                           ScenePartitionerI& partitioner,
                           TracerLogicGeneratorI& lg,
                           const SurfaceLoaderGeneratorI& sl,
                           const CudaSystem& system)
    : logicGenerator(lg)
    , cudaSystem(system)
    , partitioner(partitioner)
    , surfaceLoaderGenerator(sl)
    , maxAccelIds(Vector2i(-1))
    , maxMatIds(Vector2i(-1))
    , baseBoundaryMatKey(HitKey::InvalidKey)
    , hitStructSize(std::numeric_limits<uint32_t>::min())
    , fileName(fileName)
    , parentPath(std::filesystem::path(fileName).parent_path().string())
    , currentTime(0.0)
    , sceneJson(nullptr)
    , baseAccelerator(nullptr, nullptr)
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
    //static constexpr uint32_t MAX_UINT32 = std::numeric_limits<uint32_t>::max();

    result.clear();
    uint32_t i = 0;
    for(const auto& jsn : array)
    {
        const nlohmann::json& ids = jsn[NodeNames::ID];
        if(!ids.is_array())
        {
            //auto r = result.emplace(jsn[NodeNames::ID], std::make_pair(i, MAX_UINT32));
            auto r = result.emplace(jsn[NodeNames::ID], std::make_pair(i, 0));
            if(!r.second)
            {
                unsigned int i = static_cast<int>(SceneError::DUPLICATE_ACCELERATOR_ID) + t;
                return static_cast<SceneError::Type>(i);
            }
        }
        else
        {
            // Partially decompose id list since it can be ranged list
            std::vector<Range<uint32_t>> ranges = SceneIO::LoadRangedNumbers<uint32_t>(ids);

            SceneError e = SceneError::OK;

            // Loop over Ranged Lists
            uint32_t j = 0;
            for(const auto& range : ranges)
            {
                // Lamda to elimtinate repetition
                auto EmplaceToResult = [&](uint32_t id, uint32_t outer, uint32_t inner)
                {
                    auto r = result.emplace(id, std::make_pair(outer, inner));
                    if(!r.second)
                    {
                        unsigned int i = static_cast<int>(SceneError::DUPLICATE_ACCELERATOR_ID) + t;
                        return static_cast<SceneError::Type>(i);
                    }
                    return SceneError::OK;
                };

                if((range.end - range.start) == 1)
                {
                    if((e = EmplaceToResult(range.start, i, j)) != SceneError::OK) return e;
                    j++;
                }
                else for(uint32_t k = range.start; k < range.end; k++)
                {
                    if((e = EmplaceToResult(k, i, j)) != SceneError::OK) return e;
                    j++;
                }
            }
        }
        i++;
    }
    return SceneError::OK;
}

SceneError GPUSceneJson::GenerateConstructionData(// Striped Listings (Striped from unsued nodes)
                                                  PrimitiveNodeList& primGroupNodes,
                                                  //
                                                  MediumNodeList& mediumGroupNodes,
                                                  TransformNodeList& transformGroupNodes,
                                                  //
                                                  MaterialNodeList& matGroupNodes,
                                                  WorkBatchList& workBatchListings,
                                                  AcceleratorBatchList& requiredAccelListings,
                                                  //
                                                  CameraNodeList& cameraGroupNodes,
                                                  LightNodeList& lightGroupNodes,
                                                  //
                                                  double time)
{
    const nlohmann::json* surfaces = nullptr;
    const nlohmann::json* primitives = nullptr;
    const nlohmann::json* materials = nullptr;
    const nlohmann::json* lights = nullptr;
    const nlohmann::json* cameras = nullptr;
    const nlohmann::json* accelerators = nullptr;
    const nlohmann::json* transforms = nullptr;
    const nlohmann::json* mediums = nullptr;
    const nlohmann::json* cameraSurfaces = nullptr;
    const nlohmann::json* lightSurfaces = nullptr;
    uint32_t identityTransformId = std::numeric_limits<uint32_t>::max();

    IndexLookup primList;
    IndexLookup materialList;
    IndexLookup acceleratorList;
    IndexLookup transformList;
    IndexLookup mediumList;
    IndexLookup lightList;
    IndexLookup cameraList;

    // Lambdas for cleaner code
    auto AttachMedium = [&](uint32_t mediumId) -> SceneError
    {
        if(auto loc = mediumList.find(mediumId); loc != mediumList.end())
        {
            const NodeIndex nIndex = loc->second.first;
            const InnerIndex iIndex = loc->second.second;
            const auto& jsnNode = (*mediums)[nIndex];
            std::string mediumType = jsnNode[NodeNames::TYPE];

            auto& mediumSet = mediumGroupNodes.emplace(mediumType, NodeListing()).first->second;
            auto& node = *mediumSet.emplace(std::make_unique<SceneNodeJson>(jsnNode, nIndex)).first;
            node->AddIdIndexPair(mediumId, iIndex);
        }
        else return SceneError::MEDIUM_ID_NOT_FOUND;
        return SceneError::OK;
    };
    auto AttachWorkBatch = [&](const std::string& primType,
                               const std::string& matType,
                               const NodeId matId)
    {
        // Generate its mat batch also
        WorkBatchData batchData = WorkBatchData
        {
            primType,
            matType,
            std::set<NodeId>()
        };

        const std::string batchName = MangledNames::WorkBatch(primType.c_str(),
                                                              matType.c_str());
        const auto& workBatch = workBatchListings.emplace(batchName, batchData).first;
        workBatch->second.matIds.emplace(matId);
    };
    auto AttachMatAll = [&] (const std::string& primType,
                             const NodeId matId) -> SceneError
    {
        if(auto loc = materialList.find(matId); loc != materialList.end())
        {
            const NodeIndex nIndex = loc->second.first;
            const InnerIndex iIndex = loc->second.second;
            const auto& jsnNode = (*materials)[nIndex];

            std::string matName = jsnNode[NodeNames::TYPE];
            const std::string matGroupType = matName;
            auto& matSet = matGroupNodes.emplace(matGroupType, NodeListing()).first->second;
            auto& node = *matSet.emplace(std::make_unique<SceneNodeJson>(jsnNode, nIndex)).first;
            node->AddIdIndexPair(matId, iIndex);
            AttachWorkBatch(primType, matName, matId);

            
            // Check if this mat requires a medium
            // add to the medium list if available
            if(!node->CheckNode(NodeNames::MEDIUM)) return SceneError::OK;
            std::vector<uint32_t> mediumIds = node->AccessUInt(NodeNames::MEDIUM);
            uint32_t mediumId = mediumIds[iIndex];

            SceneError e = SceneError::OK;
            if((e = AttachMedium(mediumId)) != SceneError::OK)
                return e;

        }
        else return SceneError::MATERIAL_ID_NOT_FOUND;
        return SceneError::OK;
    };

    auto AttachTransform = [&](uint32_t transformId)
    {
        // Add Transform Group to generation list
        if(auto loc = transformList.find(transformId); loc != transformList.end())
        {
            const NodeIndex nIndex = loc->second.first;
            const InnerIndex iIndex = loc->second.second;
            const auto& jsnNode = (*transforms)[nIndex];
            std::string transformType = jsnNode[NodeNames::TYPE];

            if(transformType == NodeNames::TRANSFORM_IDENTITY)
                identityTransformId = transformId;

            auto& transformSet = transformGroupNodes.emplace(transformType, NodeListing()).first->second;
            auto& node = *transformSet.emplace(std::make_unique<SceneNodeJson>(jsnNode, nIndex)).first;
            node->AddIdIndexPair(transformId, iIndex);
        }
        else return SceneError::TRANSFORM_ID_NOT_FOUND;
        return SceneError::OK;
    };

    auto AttachAccelerator = [&](uint32_t accId, uint32_t surfId, uint32_t transformId,
                                 const std::string& primGroupType,
                                 const IdPairs& idPairs)
    {
        NodeIndex accIndex;
        std::string accType = "";
        const nlohmann::json* accNode = nullptr;

        //const uint32_t accId = surf.acceleratorId;
        if(auto loc = acceleratorList.find(accId); loc != acceleratorList.end())
        {
            accIndex = loc->second.first;
            accNode = &(*accelerators)[accIndex];
            accType = (*accNode)[NodeNames::TYPE];
        }
        else return SceneError::ACCELERATOR_ID_NOT_FOUND;
        const std::string acceleratorGroupType = MangledNames::AcceleratorGroup(primGroupType.c_str(),
                                                                                accType.c_str());
        AccelGroupData accGData =
        {
            acceleratorGroupType,
            primGroupType,
            std::map<uint32_t, IdPairs>(),
            std::vector<uint32_t>(),
            std::make_unique<SceneNodeJson>(*accNode, accIndex)
        };
        const auto& result = requiredAccelListings.emplace(acceleratorGroupType,
                                                           std::move(accGData)).first;
        result->second.matPrimIdPairs.emplace(surfId, idPairs);
        result->second.transformIds.emplace_back(transformId);
        return SceneError::OK;
    };

    // Function Start
    SceneError e = SceneError::OK;

    // Load Id Based Arrays
    if(!FindNode(cameraSurfaces, NodeNames::CAMERA_SURFACE_BASE)) return SceneError::CAMERA_SURFACES_ARRAY_NOT_FOUND;
    if(!FindNode(lightSurfaces, NodeNames::LIGHT_SURFACE_BASE)) return SceneError::LIGHT_SURFACES_ARRAY_NOT_FOUND;
    if(!FindNode(surfaces, NodeNames::SURFACE_BASE)) return SceneError::SURFACES_ARRAY_NOT_FOUND;
    if(!FindNode(primitives, NodeNames::PRIMITIVE_BASE)) return SceneError::PRIMITIVES_ARRAY_NOT_FOUND;
    if(!FindNode(materials, NodeNames::MATERIAL_BASE)) return SceneError::MATERIALS_ARRAY_NOT_FOUND;
    if(!FindNode(lights, NodeNames::LIGHT_BASE)) return SceneError::LIGHTS_ARRAY_NOT_FOUND;
    if(!FindNode(cameras, NodeNames::CAMERA_BASE)) return SceneError::CAMERAS_ARRAY_NOT_FOUND;
    if(!FindNode(accelerators, NodeNames::ACCELERATOR_BASE)) return SceneError::ACCELERATORS_ARRAY_NOT_FOUND;
    if(!FindNode(transforms, NodeNames::TRANSFORM_BASE)) return SceneError::TRANSFORMS_ARRAY_NOT_FOUND;
    if(!FindNode(mediums, NodeNames::MEDIUM_BASE)) return SceneError::MEDIUM_ARRAY_NOT_FOUND;
    if((e = GenIdLookup(primList, *primitives, PRIMITIVE)) != SceneError::OK) 
        return e;
    if((e = GenIdLookup(materialList, *materials, MATERIAL)) != SceneError::OK)
        return e;
    if((e = GenIdLookup(acceleratorList, *accelerators, ACCELERATOR)) != SceneError::OK)
        return e;
    if((e = GenIdLookup(transformList, *transforms, TRANSFORM)) != SceneError::OK)
        return e;
    if((e = GenIdLookup(mediumList, *mediums, MEDIUM)) != SceneError::OK)
        return e;
    if((e = GenIdLookup(lightList, *lights, LIGHT)) != SceneError::OK)
        return e;
    if((e = GenIdLookup(cameraList, *cameras, CAMERA)) != SceneError::OK)
        return e;

    // Iterate over surfaces
    // and collect data for groups and batches
    uint32_t surfId = 0;
    for(const auto& jsn : (*surfaces))
    {
        SurfaceStruct surf = SceneIO::LoadSurface(jsn);
        const uint32_t transformId = surf.transformId;
        
        // Start loading mats and surface data
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

                std::string currentType = jsnNode[NodeNames::TYPE];

                // All surface primitives must be same
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

        // Add Transform Group to generation list
        if((e = AttachTransform(transformId)) != SceneError::OK)
            return e;

        // Find Accelerator &
        // Add Accelerator Group to generation list
        if((e = AttachAccelerator(surf.acceleratorId, surfId, transformId,
                                  primGroupType, surf.matPrimPairs)) != SceneError::OK)
            return e;

        // Generate transform pair also
        surfId++;
    }
    // Additionally For Lights and Base Boundary Material 
    // Generate a Empty primitive (if not already requrested)
    primGroupNodes.emplace(BaseConstants::EMPTY_PRIMITIVE_NAME, NodeListing());
    
    // Boundary Material
    const nlohmann::json* baseMatNode = nullptr;
    if(!FindNode(baseMatNode, NodeNames::BASE_OUTSIDE_MATERIAL))
        return SceneError::BASE_BOUND_MAT_NODE_NOT_FOUND;
    if((e = AttachMatAll(BaseConstants::EMPTY_PRIMITIVE_NAME,
                         SceneIO::LoadNumber<uint32_t>(*baseMatNode, time))) != SceneError::OK)
        return e;
    
    // Find the base medium and tag its index
    const nlohmann::json* baseMediumNode = nullptr;
    if(!FindNode(baseMediumNode, NodeNames::BASE_MEDIUM))
        return SceneError::BASE_MEDIUM_NODE_NOT_FOUND;
    uint32_t baseMediumId = SceneIO::LoadNumber<uint32_t>(*baseMediumNode, time);    
    if((e = AttachMedium(baseMediumId)) != SceneError::OK)
        return e;
       
    // And finally force load Identity transform
    if((transformGroupNodes.find(NodeNames::TRANSFORM_IDENTITY)) == transformGroupNodes.cend())
    {
        // Assign an Unused ID
        constexpr uint32_t MAX_UINT = std::numeric_limits<uint32_t>::max();
        auto& transformSet = transformGroupNodes.emplace(NodeNames::TRANSFORM_IDENTITY, NodeListing()).first->second;
        auto& node = *transformSet.emplace(std::make_unique<SceneNodeJson>(nullptr, MAX_UINT)).first;        
        node->AddIdIndexPair(MAX_UINT, 0);
    }

    // Process Lights
    for(const auto& jsn : (*lightSurfaces))
    {
        LightSurfaceStruct s = SceneIO::LoadLightSurface(baseMediumId,
                                                         identityTransformId,
                                                         jsn);

        // Fetch type name
        std::string primTypeName = BaseConstants::EMPTY_PRIMITIVE_NAME;
        if(s.isPrimitive)
        {
            // Find Prim Type
            // And attach primitive
            if(auto loc = primList.find(s.lightOrPrimId); loc != primList.end())
            {
                const NodeIndex nIndex = loc->second.first;
                const InnerIndex iIndex = loc->second.second;
                const auto& jsnNode = (*primitives)[nIndex];

                primTypeName = jsnNode[NodeNames::TYPE];

                auto& primSet = primGroupNodes.emplace(primTypeName, NodeListing()).first->second;
                auto& node = *primSet.emplace(std::make_unique<SceneNodeJson>(jsnNode, nIndex)).first;
                node->AddIdIndexPair(s.lightOrPrimId, iIndex);
            }
            else return SceneError::PRIMITIVE_ID_NOT_FOUND;

            // Create Pairs for this object
            IdPairs pairs;
            pairs.fill(std::make_pair(std::numeric_limits<uint32_t>::max(),
                                      std::numeric_limits<uint32_t>::max()));
            pairs[0].first = s.materialId;
            pairs[0].second = s.lightOrPrimId;

            // Find Accelerator &
            // Add Accelerator Group to generation list
            if((e = AttachAccelerator(s.acceleratorId, surfId, s.transformId,
                                      primTypeName, pairs)) != SceneError::OK)
                return e;
            surfId++;
        }

        // Add Transform Group to generation list
        if((e = AttachTransform(s.transformId)) != SceneError::OK)
            return e;

        // Request material for loading
        if((e = AttachMatAll(primTypeName, s.materialId)) != SceneError::OK)
            return e;

        // Finally add to required lights
        std::unique_ptr<SceneNodeI> lightNode = nullptr;
        if(!s.isPrimitive)
        {
            // Find the light node
            if(auto loc = lightList.find(s.lightOrPrimId); loc != lightList.end())
            {
                const NodeIndex nIndex = loc->second.first;
                const InnerIndex iIndex = loc->second.second;
                const auto& jsnNode = (*lights)[nIndex];

                lightNode = std::make_unique<SceneNodeJson>(jsnNode, nIndex);
                lightNode->AddIdIndexPair(s.lightOrPrimId, iIndex);
            }
            else return SceneError::LIGHT_ID_NOT_FOUND;
        }

        // Emplace to the list
        LightGroupData data =
        {
            s.isPrimitive,
            (s.isPrimitive) ? primTypeName : "",
            std::vector<ConstructionData>()
        };
        auto& lightData = lightGroupNodes.emplace(primTypeName, std::move(data)).first->second;
        auto& constructionInfo = lightData.constructionInfo;
        constructionInfo.emplace_back(ConstructionData
                                      {
                                          s.transformId,
                                          s.mediumId,
                                          s.lightOrPrimId,
                                          s.materialId,
                                          std::move(lightNode)
                                      });

    }

    // Process Cameras
    for(const auto& jsn : (*cameraSurfaces))
    { 
        CameraSurfaceStruct s = SceneIO::LoadCameraSurface(baseMediumId,
                                                           identityTransformId,
                                                           jsn);

        // Find the light node
        std::string camTypeName;
        std::unique_ptr<SceneNodeI> cameraNode = nullptr;
        if(auto loc = cameraList.find(s.cameraId); loc != cameraList.end())
        {
            const NodeIndex nIndex = loc->second.first;
            const InnerIndex iIndex = loc->second.second;
            const auto& jsnNode = (*cameras)[nIndex];

            camTypeName = jsnNode[NodeNames::TYPE];

            cameraNode = std::make_unique<SceneNodeJson>(jsnNode, nIndex);
            cameraNode->AddIdIndexPair(s.cameraId, iIndex);
        }
        else return SceneError::CAMERA_ID_NOT_FOUND;

        // Emplace to the list
        auto& camConstructionInfo = cameraGroupNodes.emplace(camTypeName, std::vector<ConstructionData>()).first->second;
        camConstructionInfo.emplace_back(ConstructionData
                                         {
                                             s.transformId,
                                             s.mediumId,
                                             s.cameraId,
                                             s.materialId,
                                             std::move(cameraNode)
                                         });

    }
    return e;
}

SceneError GPUSceneJson::GenerateMaterialGroups(const MultiGPUMatNodes& matGroupNodes,
                                                const std::map<uint32_t, uint32_t>& mediumIdMappings,
                                                double time)
{
    // Generate Partitioned Material Groups
    SceneError e = SceneError::OK;
    for(const auto& matGroupN : matGroupNodes)
    {
        const std::string& matTypeName = matGroupN.first.first;
        const CudaGPU* gpu = matGroupN.first.second;
        const auto& matNodes = matGroupN.second;
        //
        GPUMatGPtr matGroup = GPUMatGPtr(nullptr, nullptr);
        if(e = logicGenerator.GenerateMaterialGroup(matGroup, *gpu, matTypeName))
            return e;
        if(e = matGroup->InitializeGroup(matNodes, mediumIdMappings, time, parentPath))
            return e;
        materials.emplace(std::make_pair(matTypeName, gpu), std::move(matGroup));
    }
    return e;
}

SceneError GPUSceneJson::GenerateWorkBatches(MaterialKeyListing& allMatKeys,
                                             const MultiGPUWorkBatches& materialBatches,
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

        // Generate Keys
        const CudaGPU* gpu = requiredMat.first.second;
        const std::string& matTName = requiredMat.second.matType;
        const std::string& primTName = requiredMat.second.primType;
        // Find Interfaces
        // and generate work info
        GPUPrimitiveGroupI* pGroup = primitives.at(primTName).get();
        GPUMaterialGroupI* mGroup = materials.at(std::make_pair(matTName, gpu)).get();
        workInfo.emplace_back(batchId, pGroup, mGroup);

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
        GPUPrimGPtr pg = GPUPrimGPtr(nullptr, nullptr);
        if(e = logicGenerator.GeneratePrimitiveGroup(pg, primTypeName))
            return e;
        if(e = pg->InitializeGroup(primNodes, time, surfaceLoaderGenerator, parentPath))
            return e;

        ExpandHitStructSize(*pg.get());
        primitives.emplace(primTypeName, std::move(pg));
    }
    return e;
}

SceneError GPUSceneJson::GenerateAccelerators(std::map<uint32_t, HitKey>& accHitKeyList,
                                              //
                                              const AcceleratorBatchList& acceleratorBatchList,
                                              const MaterialKeyListing& matHitKeyList,
                                              //
                                              const std::map<uint32_t, uint32_t>& transformIdMappings,
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
        const auto& accelNode = accelGroupBatch.second.accelNode;

        // Convert TransformIdList to transformIndexList
        auto& transformIdList = accelGroupBatch.second.transformIds;

        std::vector<uint32_t> transformIndexList;
        transformIndexList.resize(transformIdList.size());
        std::transform(std::execution::par_unseq,
                       transformIdList.cbegin(), transformIdList.cend(),
                       transformIndexList.begin(),
        [&indexLookup = std::as_const(transformIdMappings)](uint32_t id)
        {
            // Convert id to index
            return indexLookup.at(id);
        });

        // Fetch Primitive
        GPUPrimitiveGroupI* pGroup = primitives.at(primTName).get();

        // Group Generation
        GPUAccelGPtr aGroup = GPUAccelGPtr(nullptr, nullptr);
        if((e = logicGenerator.GenerateAcceleratorGroup(aGroup, *pGroup, accelGroupName)) != SceneError::OK)
            return e;
        if((e = aGroup->InitializeGroup(accelNode, matHitKeyList, 
                                        pairsList, transformIndexList,
                                        time)) != SceneError::OK)
            return e;

        // Batch Generation
        accelMap.emplace(accelId, aGroup.get());

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

        // Finally emplace it to the list
        accelerators.emplace(accelGroupName, std::move(aGroup));
    }
    return e;
}

SceneError GPUSceneJson::GenerateBaseAccelerator(const std::map<uint32_t, HitKey>& accHitKeyList,
                                                 double time)
{
    SceneError e = SceneError::OK;

    // Find Base Accelerator Type and generate
    const nlohmann::json* baseAccel = nullptr;
    if(!FindNode(baseAccel, NodeNames::BASE_ACCELERATOR))
        return SceneError::BASE_ACCELERATOR_NODE_NOT_FOUND;
    const std::string baseAccelType = (*baseAccel)[NodeNames::TYPE];

    // Generate Base Accelerator
    baseAccelerator = nullptr;
    if((e = logicGenerator.GenerateBaseAccelerator(baseAccelerator, baseAccelType)) != SceneError::OK)
        return e;
    if((e = baseAccelerator->Initialize(std::make_unique<SceneNodeJson>(*baseAccel, 0),
                                        accHitKeyList)) != SceneError::OK)
        return e;
    return e;
}

SceneError GPUSceneJson::GenerateTransforms(std::map<uint32_t, uint32_t>& transformIdMappings,
                                            uint32_t& identityTIndex,
                                            const TransformNodeList& transformList,
                                            double time)
{
    // Generate Transform Groups
    SceneError e = SceneError::OK;
    uint32_t linearIndex = 0;
    bool hadIdentityTransform = false;
    for(const auto& transformGroup : transformList)
    {
        std::string transTypeName = transformGroup.first;
        const auto& transNodes = transformGroup.second;
        //
        CPUTransformGPtr tg = CPUTransformGPtr(nullptr, nullptr);
        if(e = logicGenerator.GenerateTransformGroup(tg, transTypeName))
            return e;
        if(e = tg->InitializeGroup(transNodes, time, parentPath))
            return e;
        transforms.emplace(transTypeName, std::move(tg));

        for(const auto& node : transNodes)
        for(const auto& idPair : node->Ids())
        {
            uint32_t sceneTransId = idPair.first;
            transformIdMappings.emplace(sceneTransId, linearIndex);

            // Set Identity Transform Index
            if(transTypeName == std::string(NodeNames::TRANSFORM_IDENTITY))
                identityTransformIndex = linearIndex;

            linearIndex++;
        }
    }
    return e;
}

SceneError GPUSceneJson::GenerateMediums(std::map<uint32_t, uint32_t>& mediumIdMappings,
                                         uint32_t& baseMIndex,
                                         const MediumNodeList& mediumList,
                                         double time)
{
    // Find the base medium and tag its index
    const nlohmann::json* baseMediumNode = nullptr;
    if(!FindNode(baseMediumNode, NodeNames::BASE_MEDIUM))
        return SceneError::BASE_MEDIUM_NODE_NOT_FOUND;
    uint32_t baseMediumId = SceneIO::LoadNumber<uint32_t>(*baseMediumNode, time);

    // Generate Transform Groups
    uint32_t linearIndex = 0;
    bool baseMediumFound = false;
    SceneError e = SceneError::OK;
    for(const auto& mediumGroup : mediumList)
    {
        std::string mediumTypeName = mediumGroup.first;
        const auto& mediumNodes = mediumGroup.second;
        //
        CPUMediumGPtr mg = CPUMediumGPtr(nullptr, nullptr);
        if(e = logicGenerator.GenerateMediumGroup(mg, mediumTypeName))
            return e;
        if(e = mg->InitializeGroup(mediumNodes, time, parentPath))
            return e;
        mediums.emplace(mediumTypeName, std::move(mg));

        for(const auto& node : mediumNodes)
        for(const auto& idPair : node->Ids())
        {
            uint32_t sceneMedId = idPair.first;

            mediumIdMappings.emplace(sceneMedId, linearIndex);
            if(baseMediumId == sceneMedId)
            {
                baseMediumIndex = linearIndex;
                baseMediumFound = true;
            }
            linearIndex++;
        }
    }
    
    if(!baseMediumFound) 
        return SceneError::MEDIUM_ID_NOT_FOUND;
    return e;
}

SceneError GPUSceneJson::GenerateCameras(const CameraNodeList& camGroupList,
                                         const std::map<uint32_t, uint32_t>& transformIdMappings,
                                         const std::map<uint32_t, uint32_t>& mediumIdMappings,
                                         const MaterialKeyListing& materialKeys,
                                         double time)
{
    SceneError e = SceneError::OK;
    for(const auto& camGroup : camGroupList)
    {
        const std::string& camTypeName = camGroup.first;
        const auto& camNodes = camGroup.second;

        CPUCameraGPtr cg = CPUCameraGPtr(nullptr, nullptr);
        if(e = logicGenerator.GenerateCameraGroup(cg, camTypeName))
            return e;
        if(e = cg->InitializeGroup(camNodes,
                                   mediumIdMappings,
                                   transformIdMappings,
                                   materialKeys,
                                   time,
                                   parentPath))
            return e;
        cameras.emplace(camTypeName, std::move(cg));
    }
    return SceneError::OK;
}

SceneError GPUSceneJson::GenerateLights(const LightNodeList& lightGroupList,
                                        const std::map<uint32_t, uint32_t>& transformIdMappings,
                                        const std::map<uint32_t, uint32_t>& mediumIdMappings,
                                        const MaterialKeyListing& materialKeys,
                                        double time)
{
    SceneError e = SceneError::OK;
    for(const auto& lightGroup : lightGroupList)
    {
        const std::string& lightTypeName = lightGroup.first;
        const std::string& primTypeName = lightGroup.second.primTypeName;
        bool isPrimLight = lightGroup.second.isPrimitive;
        const auto& lightNodes = lightGroup.second.constructionInfo;

        // Find Primitive
        GPUPrimitiveGroupI* primGroup = nullptr;
        if(isPrimLight)
            primGroup = primitives.at(primTypeName).get();

        CPULightGPtr lg = CPULightGPtr(nullptr, nullptr);
        if(e = logicGenerator.GenerateLightGroup(lg, primGroup, lightTypeName))
            return e;
        if(e = lg->InitializeGroup(lightNodes,
                                   mediumIdMappings,
                                   transformIdMappings,
                                   materialKeys,
                                   time,
                                   parentPath))
            return e;
        lights.emplace(lightTypeName, std::move(lg));
    }
    return SceneError::OK;
}

SceneError GPUSceneJson::FindBoundaryMaterial(const MaterialKeyListing& matHitKeyList,
                                              double time)
{
    SceneError e = SceneError::OK;
    NodeListing nodeList;

    const nlohmann::json* node = nullptr;
    if(!FindNode(node, NodeNames::BASE_OUTSIDE_MATERIAL))
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

SceneError GPUSceneJson::LoadAll(double time)
{
    SceneError e = SceneError::OK;
    // Group Data
    PrimitiveNodeList primGroupNodes;
    //
    MediumNodeList mediumGroupNodes;
    TransformNodeList transformGroupNodes;
    MaterialNodeList matGroupNodes;
    WorkBatchList workListings;
    AcceleratorBatchList accelListings;
    CameraNodeList camListings;
    LightNodeList lightListings;
    // Parse Json and find necessary nodes
    if((e = GenerateConstructionData(primGroupNodes,
                                     mediumGroupNodes,
                                     transformGroupNodes,
                                     matGroupNodes,
                                     workListings,
                                     accelListings,
                                     camListings,
                                     lightListings,
                                     time)) != SceneError::OK)
        return e;

    // Transforms
    std::map<uint32_t, uint32_t> transformIdMappings;
    if((e = GenerateTransforms(transformIdMappings, identityTransformIndex,
                               transformGroupNodes,
                               time)) != SceneError::OK)
        return e;

    // Partition Material Data to Multi GPU Material Data
    MultiGPUMatNodes multiGPUMatNodes;
    MultiGPUWorkBatches multiGPUWorkBatches;
    if((e = partitioner.PartitionMaterials(multiGPUMatNodes,
                                           multiGPUWorkBatches,
                                           //
                                           matGroupNodes,
                                           workListings)))
        return e;

    // Mediums
    std::map<uint32_t, uint32_t> mediumIdMappings;
    if((e = GenerateMediums(mediumIdMappings, baseMediumIndex,
                            mediumGroupNodes,
                            time)) != SceneError::OK)
        return e;

    // Using those constructs generate
    // Primitive Groups
    if((e = GeneratePrimitiveGroups(primGroupNodes, time)) != SceneError::OK)
        return e;
    // Material Groups
    if((e = GenerateMaterialGroups(multiGPUMatNodes, mediumIdMappings, time)) != SceneError::OK)
        return e;
    // Work Batches
    MaterialKeyListing allMaterialKeys;
    if((e = GenerateWorkBatches(allMaterialKeys,
                                multiGPUWorkBatches,
                                time)) != SceneError::OK)
        return e;
    // Accelerators
    std::map<uint32_t, HitKey> accHitKeyList;
    if((e = GenerateAccelerators(accHitKeyList, accelListings,
                                 allMaterialKeys, transformIdMappings,
                                 time)) != SceneError::OK)
        return e;
    // Base Accelerator
    if((e = GenerateBaseAccelerator(accHitKeyList, time)) != SceneError::OK)
        return e;
    // Boundary Material
    if((e = FindBoundaryMaterial(allMaterialKeys, time)) != SceneError::OK)
       return e;    
    // Cameras
    if((e = GenerateCameras(camListings,
                            transformIdMappings,
                            mediumIdMappings,
                            allMaterialKeys, 
                            time)) != SceneError::OK)
        return e;
    // Lights
    if((e = GenerateLights(lightListings,
                           transformIdMappings,
                           mediumIdMappings,
                           allMaterialKeys,
                           time)) != SceneError::OK)
        return e;

    // MaxIds are generated but those are inclusive
    // Make them exclusve
    maxAccelIds += Vector2i(1);
    maxMatIds += Vector2i(1);

    // Everything is generated!
    return SceneError::OK;
}

SceneError GPUSceneJson::ChangeAll(double time)
{
    // TODO:
    return SceneError::SURFACE_LOADER_INTERNAL_ERROR;
}

size_t GPUSceneJson::UsedGPUMemory() const
{
    return 0;
}

size_t GPUSceneJson::UsedCPUMemory() const
{
    return 0;
}

SceneError GPUSceneJson::LoadScene(double time)
{
    SceneError e = SceneError::OK;
    try
    {
        if((e = OpenFile(fileName)) != SceneError::OK)
           return e;
        if((e = LoadAll(time)) != SceneError::OK)
            return e;
    }
    catch (SceneException const& e)
    {
        return e;
    }
    catch(nlohmann::json::parse_error const& e)
    {
        METU_ERROR_LOG("%s", e.what());
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
        if((e = ChangeAll(time)) != SceneError::OK)
            return e;
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

Vector2i GPUSceneJson::MaxMatIds() const
{
    return maxMatIds;
}

Vector2i GPUSceneJson::MaxAccelIds() const
{
    return maxAccelIds;
}

HitKey GPUSceneJson::BaseBoundaryMaterial() const
{
    return baseBoundaryMatKey;
}

uint32_t GPUSceneJson::HitStructUnionSize() const
{
    return hitStructSize;
}

const NamedList<CPULightGPtr>& GPUSceneJson::Lights() const
{
    return lights;
}

const NamedList<CPUCameraGPtr>& GPUSceneJson::Cameras() const
{
    return cameras;
}

const NamedList<CPUTransformGPtr>& GPUSceneJson::Transforms() const
{
    return transforms;
}

const NamedList<CPUMediumGPtr>& GPUSceneJson::Mediums() const
{
    return mediums;
}

uint32_t GPUSceneJson::BaseMediumIndex() const
{
    return baseMediumIndex;
}

uint32_t GPUSceneJson::IdentityTransformIndex() const
{
    return identityTransformIndex;
}

const WorkBatchCreationInfo& GPUSceneJson::WorkBatchInfo() const
{
    return workInfo;
}

const AcceleratorBatchMap& GPUSceneJson::AcceleratorBatchMappings() const
{
    return accelMap;
}

const GPUBaseAccelPtr& GPUSceneJson::BaseAccelerator() const
{
    return baseAccelerator;
}

const std::map<NameGPUPair, GPUMatGPtr>& GPUSceneJson::MaterialGroups() const
{
    return materials;
}

const NamedList<GPUAccelGPtr>& GPUSceneJson::AcceleratorGroups() const
{
    return accelerators;
}

const NamedList<GPUPrimGPtr>& GPUSceneJson::PrimitiveGroups() const
{
    return primitives;
}