#include "GPUSceneJson.h"

#include "RayLib/SceneIO.h"
#include "RayLib/Types.h"
#include "RayLib/Log.h"
#include "RayLib/SceneNodeI.h"
#include "RayLib/SceneNodeNames.h"
#include "RayLib/StripComments.h"
#include "RayLib/FileSystemUtility.h"
#include "RayLib/UTF8StringConversion.h"
#include "RayLib/AnalyticData.h"
#include "RayLib/CPUTimer.h"

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

SceneError GPUSceneJson::OpenFile(const std::u8string& filePath)
{
    // TODO: get a lightweight lexer and strip comments
    // from json since json does not support comments
    // now its only pure json iterating over a scene is
    // not convenient without comments.

    // Always assume filenames are UTF-8
    const auto path = std::filesystem::path(filePath);
    std::ifstream file(path);

    if(!file.is_open()) return SceneError(SceneError::FILE_NOT_FOUND, path.generic_string());
    // Parse Json
    sceneJson = std::make_unique<nlohmann::json>();
    (*sceneJson) = nlohmann::json::parse(file, nullptr, true, true);

    return SceneError::OK;
}

GPUSceneJson::GPUSceneJson(const std::u8string& fileName,
                           ScenePartitionerI& partitioner,
                           TracerLogicGeneratorI& lg,
                           const SurfaceLoaderGeneratorI& sl,
                           const CudaSystem& system,
                           SceneLoadFlags flags)
    : logicGenerator(lg)
    , partitioner(partitioner)
    , surfaceLoaderGenerator(sl)
    , cudaSystem(system)
    , loadFlags(flags)

    , maxAccelIds(Vector2i(-1))
    , maxMatIds(Vector2i(-1))
    , baseBoundaryMatKey(HitKey::InvalidKey)
    , hitStructSize(std::numeric_limits<uint32_t>::min())
    , identityTransformIndex(std::numeric_limits<uint32_t>::max())
    , boundaryTransformIndex(std::numeric_limits<uint32_t>::max())
    , baseMediumIndex(std::numeric_limits<uint16_t>::max())
    , cameraCount(0)
    , baseAccelerator(nullptr, nullptr)
    , sceneJson(nullptr)
    , fileName(fileName)
    , parentPath(std::filesystem::path(fileName).parent_path().string())
    , currentTime(0.0)
{}

bool GPUSceneJson::FindNode(const nlohmann::json*& jsn, const char* c)
{
    const auto i = sceneJson->find(c);
    bool found = (i != sceneJson->end());
    if(found) jsn = &(*i);
    return found;
}

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
            uint32_t id = jsn[NodeNames::ID];
            auto r = result.emplace(id, std::make_pair(i, 0));
            if(!r.second)
            {
                unsigned int scnErrInt = static_cast<int>(SceneError::DUPLICATE_ACCELERATOR_ID) + t;
                return SceneError(static_cast<SceneError::Type>(scnErrInt),
                                  std::to_string(id));
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
                // Lambda to eliminate repetition
                auto EmplaceToResult = [&](uint32_t id, uint32_t outer, uint32_t inner) -> SceneError
                {
                    auto r = result.emplace(id, std::make_pair(outer, inner));
                    if(!r.second)
                    {
                        unsigned int i = static_cast<int>(SceneError::DUPLICATE_ACCELERATOR_ID) + t;
                        return SceneError(static_cast<SceneError::Type>(i),
                                          std::to_string(id));
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

SceneError GPUSceneJson::GenerateConstructionData(// Striped Listings (Striped from unused nodes)
                                                  PrimitiveNodeList& primGroupNodes,
                                                  //
                                                  MediumNodeList& mediumGroupNodes,
                                                  TransformNodeList& transformGroupNodes,
                                                  uint32_t& boundaryTransformId,
                                                  //
                                                  std::string& bLightGroupTypeName,
                                                  uint32_t& bLightId,
                                                  //
                                                  MaterialNodeList& matGroupNodes,
                                                  WorkBatchList& workBatchListings,
                                                  AcceleratorBatchList& requiredAccelListings,
                                                  //
                                                  CameraNodeList& cameraGroupNodes,
                                                  LightNodeList& lightGroupNodes,
                                                  //
                                                  TextureNodeMap& textureNodes,
                                                  //
                                                  double time)
{
    const nlohmann::json emptyJson;
    const nlohmann::json* surfacesJson = nullptr;
    const nlohmann::json* primitivesJson = nullptr;
    const nlohmann::json* materialsJson = nullptr;
    const nlohmann::json* lightsJson = nullptr;
    const nlohmann::json* camerasJson = nullptr;
    const nlohmann::json* acceleratorsJson = nullptr;
    const nlohmann::json* transformsJson = nullptr;
    const nlohmann::json* mediumsJson = nullptr;
    const nlohmann::json* cameraSurfacesJson = nullptr;
    const nlohmann::json* lightSurfacesJson = nullptr;
    const nlohmann::json* texturesJson = nullptr;
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
            const auto& jsnNode = (*mediumsJson)[nIndex];
            std::string mediumType = jsnNode[NodeNames::TYPE];

            auto& mediumSet = mediumGroupNodes.emplace(mediumType, NodeListing()).first->second;
            auto& node = *mediumSet.emplace(std::make_unique<SceneNodeJson>(jsnNode, nIndex)).first;
            node->AddIdIndexPair(mediumId, iIndex);
        }
        else return SceneError(SceneError::MEDIUM_ID_NOT_FOUND, std::to_string(mediumId));
        return SceneError::OK;
    };
    auto AttachWorkBatch = [&](const std::string& primType,
                               const std::string& matType,
                               const NodeId matId) -> void
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
            const auto& jsnNode = (*materialsJson)[nIndex];

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
        else return SceneError(SceneError::MATERIAL_ID_NOT_FOUND, std::to_string(matId));
        return SceneError::OK;
    };
    auto AttachTransform = [&](uint32_t transformId) -> SceneError
    {
        // Add Transform Group to generation list
        if(auto loc = transformList.find(transformId); loc != transformList.end())
        {
            const NodeIndex nIndex = loc->second.first;
            const InnerIndex iIndex = loc->second.second;
            const auto& jsnNode = (*transformsJson)[nIndex];
            std::string transformType = jsnNode[NodeNames::TYPE];

            if(transformType == NodeNames::TRANSFORM_IDENTITY)
                identityTransformId = transformId;

            auto& transformSet = transformGroupNodes.emplace(transformType, NodeListing()).first->second;
            auto& node = *transformSet.emplace(std::make_unique<SceneNodeJson>(jsnNode, nIndex)).first;
            node->AddIdIndexPair(transformId, iIndex);
        }
        else return SceneError(SceneError::TRANSFORM_ID_NOT_FOUND, std::to_string(transformId));
        return SceneError::OK;
    };
    auto AttachAccelerator = [&](uint32_t accId, uint32_t surfId, uint32_t transformId,
                                 const std::string& primGroupType,
                                 const IdPairs& idPairs) -> SceneError
    {
        NodeIndex accIndex;
        std::string accType = "";
        const nlohmann::json* accNode = nullptr;

        if(auto loc = acceleratorList.find(accId); loc != acceleratorList.end())
        {
            accIndex = loc->second.first;
            accNode = &(*acceleratorsJson)[accIndex];
            accType = (std::as_const(loadFlags)[SceneLoadFlagType::FORCE_OPTIX_ACCELS])
                        ? SceneConstants::OptiXAcceleratorTypeName
                        : std::string((*accNode)[NodeNames::TYPE]);
        }
        else return SceneError(SceneError::ACCELERATOR_ID_NOT_FOUND, std::to_string(accId));
        const std::string acceleratorGroupType = MangledNames::AcceleratorGroup(primGroupType.c_str(),
                                                                                accType.c_str());
        AccelGroupData accGData =
        {
            acceleratorGroupType,
            primGroupType,
            std::map<uint32_t, AccelGroupData::SurfaceDef>(),
            std::map<uint32_t, AccelGroupData::LSurfaceDef>(),
            std::make_unique<SceneNodeJson>(*accNode, accIndex)
        };
        const auto& result = requiredAccelListings.emplace(acceleratorGroupType,
                                                           std::move(accGData)).first;
        result->second.surfaces.emplace(surfId, AccelGroupData::SurfaceDef{transformId, idPairs});
        return SceneError::OK;
    };
    auto AttachAcceleratorForLight = [&](uint32_t accId, uint32_t surfId,
                                         uint32_t transformId, uint32_t lightId,
                                         const std::string& primGroupType,
                                         const uint32_t primitiveId) -> SceneError
    {
        NodeIndex accIndex;
        std::string accType = "";
        const nlohmann::json* accNode = nullptr;

        //const uint32_t accId = surf.acceleratorId;
        if(auto loc = acceleratorList.find(accId); loc != acceleratorList.end())
        {
            accIndex = loc->second.first;
            accNode = &(*acceleratorsJson)[accIndex];
            accType = (std::as_const(loadFlags)[SceneLoadFlagType::FORCE_OPTIX_ACCELS])
                        ? SceneConstants::OptiXAcceleratorTypeName
                        : std::string((*accNode)[NodeNames::TYPE]);
        }
        else return SceneError(SceneError::ACCELERATOR_ID_NOT_FOUND, std::to_string(accId));
        const std::string acceleratorGroupType = MangledNames::AcceleratorGroup(primGroupType.c_str(),
                                                                                accType.c_str());
        AccelGroupData accGData =
        {
            acceleratorGroupType,
            primGroupType,
            std::map<uint32_t, AccelGroupData::SurfaceDef>(),
            std::map<uint32_t, AccelGroupData::LSurfaceDef>(),
            std::make_unique<SceneNodeJson>(*accNode, accIndex)
        };
        const auto& result = requiredAccelListings.emplace(acceleratorGroupType,
                                                           std::move(accGData)).first;
        result->second.lightSurfaces.emplace(surfId, AccelGroupData::LSurfaceDef{transformId, primitiveId, lightId});
        return SceneError::OK;
    };

    // Function Start
    SceneError e = SceneError::OK;

    // Load Id Based Arrays
    if(!FindNode(cameraSurfacesJson, NodeNames::CAMERA_SURFACE_BASE)) return SceneError::CAMERA_SURFACES_ARRAY_NOT_FOUND;
    if(!FindNode(lightSurfacesJson, NodeNames::LIGHT_SURFACE_BASE)) return SceneError::LIGHT_SURFACES_ARRAY_NOT_FOUND;
    if(!FindNode(surfacesJson, NodeNames::SURFACE_BASE)) return SceneError::SURFACES_ARRAY_NOT_FOUND;
    if(!FindNode(primitivesJson, NodeNames::PRIMITIVE_BASE)) return SceneError::PRIMITIVES_ARRAY_NOT_FOUND;
    if(!FindNode(materialsJson, NodeNames::MATERIAL_BASE)) return SceneError::MATERIALS_ARRAY_NOT_FOUND;
    if(!FindNode(lightsJson, NodeNames::LIGHT_BASE)) return SceneError::LIGHTS_ARRAY_NOT_FOUND;
    if(!FindNode(camerasJson, NodeNames::CAMERA_BASE)) return SceneError::CAMERAS_ARRAY_NOT_FOUND;
    if(!FindNode(acceleratorsJson, NodeNames::ACCELERATOR_BASE)) return SceneError::ACCELERATORS_ARRAY_NOT_FOUND;
    if(!FindNode(transformsJson, NodeNames::TRANSFORM_BASE)) return SceneError::TRANSFORMS_ARRAY_NOT_FOUND;
    if(!FindNode(mediumsJson, NodeNames::MEDIUM_BASE)) return SceneError::MEDIUM_ARRAY_NOT_FOUND;
    if(!FindNode(texturesJson, NodeNames::TEXTURE_BASE)) texturesJson = &emptyJson;
    if((e = GenIdLookup(primList, *primitivesJson, PRIMITIVE)) != SceneError::OK)
        return e;
    if((e = GenIdLookup(materialList, *materialsJson, MATERIAL)) != SceneError::OK)
        return e;
    if((e = GenIdLookup(acceleratorList, *acceleratorsJson, ACCELERATOR)) != SceneError::OK)
        return e;
    if((e = GenIdLookup(transformList, *transformsJson, TRANSFORM)) != SceneError::OK)
        return e;
    if((e = GenIdLookup(mediumList, *mediumsJson, MEDIUM)) != SceneError::OK)
        return e;
    if((e = GenIdLookup(lightList, *lightsJson, LIGHT)) != SceneError::OK)
        return e;
    if((e = GenIdLookup(cameraList, *camerasJson, CAMERA)) != SceneError::OK)
        return e;

    // Initially find IdentityTransform
    // if available or add it on your own
    for(const auto& tNode : transformList)
    {
        const NodeIndex nIndex = tNode.second.first;
        const auto& jsnNode = (*transformsJson)[nIndex];
        std::string transformType = jsnNode[NodeNames::TYPE];
        if(transformType == NodeNames::TRANSFORM_IDENTITY)
        {
            identityTransformId = tNode.first;
            AttachTransform(identityTransformId);
        }
    }
    // If we did not found identity transform create it on your own
    if((transformGroupNodes.find(NodeNames::TRANSFORM_IDENTITY)) == transformGroupNodes.cend())
    {
        // Assign an Unused ID
        constexpr uint32_t MAX_UINT = std::numeric_limits<uint32_t>::max();
        auto& transformSet = transformGroupNodes.emplace(NodeNames::TRANSFORM_IDENTITY, NodeListing()).first->second;
        auto& node = *transformSet.emplace(std::make_unique<SceneNodeJson>(nullptr, MAX_UINT)).first;
        node->AddIdIndexPair(MAX_UINT, 0);
        identityTransformId = MAX_UINT;
    }

    // Iterate over surfaces
    // and collect data for groups and batches
    uint32_t surfId = 0;
    for(const auto& jsn : (*surfacesJson))
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
                const auto& jsnNode = (*primitivesJson)[nIndex];

                std::string currentType = jsnNode[NodeNames::TYPE];

                // All surface primitives must be same
                if((i != 0) && primGroupType != currentType)
                    return SceneError(SceneError::PRIM_TYPE_NOT_CONSISTENT_ON_SURFACE,
                                      "surface index " + std::to_string(surfId));
                else primGroupType = currentType;
                auto& primSet = primGroupNodes.emplace(primGroupType, NodeListing()).first->second;
                auto& node = *primSet.emplace(std::make_unique<SceneNodeJson>(jsnNode, nIndex)).first;
                node->AddIdIndexPair(primId, iIndex);
            }
            else return SceneError(SceneError::PRIMITIVE_ID_NOT_FOUND,
                                   std::to_string(primId));

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
    // Additionally For Lights & Cameras
    // generate an empty primitive (if not already requested)
    primGroupNodes.emplace(BaseConstants::EMPTY_PRIMITIVE_NAME, NodeListing());

    // Find the base medium and tag its index
    const nlohmann::json* baseMediumNode = nullptr;
    if(!FindNode(baseMediumNode, NodeNames::BASE_MEDIUM))
        return SceneError::BASE_MEDIUM_NODE_NOT_FOUND;
    uint32_t baseMediumId = SceneIO::LoadNumber<uint32_t>(*baseMediumNode, time);
    if((e = AttachMedium(baseMediumId)) != SceneError::OK)
        return e;

    // Find the boundary transform id if available
    // If not default to identity transform id
    const nlohmann::json* boundaryTransformNode = nullptr;
    if(!FindNode(boundaryTransformNode, NodeNames::BASE_BOUNDARY_TRANSFORM))
        boundaryTransformId = identityTransformId;
    else
    {
        boundaryTransformId = SceneIO::LoadNumber<uint32_t>(*boundaryTransformNode, time);
        if((e = AttachTransform(boundaryTransformId)) != SceneError::OK)
            return e;
    }

    // Process Lights
    for(const auto& jsn : (*lightSurfacesJson))
    {
        LightSurfaceStruct s = SceneIO::LoadLightSurface(baseMediumId,
                                                         identityTransformId,
                                                         jsn, (*lightsJson),
                                                         lightList);

        // Fetch type name
        std::string primTypeName = BaseConstants::EMPTY_PRIMITIVE_NAME;
        if(s.isPrimitive)
        {
            // Find Prim Type
            // And attach primitive
            if(auto loc = primList.find(s.primId); loc != primList.end())
            {
                const NodeIndex nIndex = loc->second.first;
                const InnerIndex iIndex = loc->second.second;
                const auto& jsnNode = (*primitivesJson)[nIndex];

                primTypeName = jsnNode[NodeNames::TYPE];

                auto& primSet = primGroupNodes.emplace(primTypeName, NodeListing()).first->second;
                auto& node = *primSet.emplace(std::make_unique<SceneNodeJson>(jsnNode, nIndex)).first;
                node->AddIdIndexPair(s.primId, iIndex);
            }
            else return SceneError(SceneError::PRIMITIVE_ID_NOT_FOUND,
                                   std::to_string(s.primId));

            // Attach Accelerator
            if((e = AttachAcceleratorForLight(s.acceleratorId, surfId,
                                              s.transformId, s.lightId,
                                      primTypeName, s.primId)) != SceneError::OK)
                return e;
        }
        else
        {
            // Only Attach medium if light is not a primitive
            if((e = AttachMedium(s.mediumId)) != SceneError::OK)
                return e;
        }

        // Add Transform Group to generation list
        if((e = AttachTransform(s.transformId)) != SceneError::OK)
            return e;

        // Finally add to required lights
        std::string lightTypeName = "";
        std::unique_ptr<SceneNodeI> lightNode = nullptr;
        // Find the light node
        if(auto loc = lightList.find(s.lightId); loc != lightList.end())
        {
            const NodeIndex nIndex = loc->second.first;
            const InnerIndex iIndex = loc->second.second;
            const auto& jsnNode = (*lightsJson)[nIndex];
            lightTypeName = jsnNode[NodeNames::TYPE];

            lightNode = std::make_unique<SceneNodeJson>(jsnNode, nIndex);
            lightNode->AddIdIndexPair(s.lightId, iIndex);
        }
        else return SceneError(SceneError::LIGHT_ID_NOT_FOUND, std::to_string(s.lightId));

        // Emplace to the list
        LightGroupData data =
        {
            primTypeName,
            std::vector<EndpointConstructionData>()
        };

        // Key of the primitive backed lights are the primitive's name
        std::string groupName = (s.isPrimitive) ? primTypeName : lightTypeName;
        groupName = MangledNames::LightGroup(groupName.c_str());
        auto& lightData = lightGroupNodes.emplace(groupName, std::move(data)).first->second;
        auto& constructionInfo = lightData.constructionInfo;
        constructionInfo.emplace_back(EndpointConstructionData
                                      {
                                          surfId,
                                          s.transformId,
                                          s.mediumId,
                                          s.lightId,
                                          s.primId,
                                          std::move(lightNode),
                                      });
        surfId++;
    }

    // Process Cameras
    cameraCount = static_cast<uint32_t>(cameraSurfacesJson->size());
    for(const auto& jsn : (*cameraSurfacesJson))
    {
        CameraSurfaceStruct s = SceneIO::LoadCameraSurface(baseMediumId,
                                                           identityTransformId,
                                                           jsn);

        // Find the camera node
        std::string camTypeName;
        std::unique_ptr<SceneNodeI> cameraNode = nullptr;
        if(auto loc = cameraList.find(s.cameraId); loc != cameraList.end())
        {
            const NodeIndex nIndex = loc->second.first;
            const InnerIndex iIndex = loc->second.second;
            const auto& jsnNode = (*camerasJson)[nIndex];

            camTypeName = jsnNode[NodeNames::TYPE];

            cameraNode = std::make_unique<SceneNodeJson>(jsnNode, nIndex);
            cameraNode->AddIdIndexPair(s.cameraId, iIndex);
        }
        else return SceneError(SceneError::CAMERA_ID_NOT_FOUND, std::to_string(s.cameraId));
        //
        if((e = AttachMedium(s.mediumId)) != SceneError::OK)
            return e;
        //
        if((e = AttachTransform(s.transformId)) != SceneError::OK)
            return e;

        // Emplace to the list
        std::string groupName = MangledNames::CameraGroup(camTypeName.c_str());
        auto& camConstructionInfo = cameraGroupNodes.emplace(groupName, std::vector<EndpointConstructionData>()).first->second;
        camConstructionInfo.emplace_back(EndpointConstructionData
                                         {
                                             surfId,
                                             s.transformId,
                                             s.mediumId,
                                             s.cameraId,
                                             std::numeric_limits<uint32_t>::max(),
                                             std::move(cameraNode)
                                         });

        surfId++;
    }

    // Process Boundary Light
    const nlohmann::json* boundLightNode = nullptr;
    if(!FindNode(boundLightNode, NodeNames::BASE_BOUNDARY_LIGHT))
        return SceneError::BASE_BOUND_LIGHT_NODE_NOT_FOUND;
    uint32_t lightId = (*boundLightNode);
    if(auto loc = lightList.find(lightId); loc != lightList.end())
    {
        const NodeIndex nIndex = loc->second.first;
        const InnerIndex iIndex = loc->second.second;
        const auto& jsnNode = (*lightsJson)[nIndex];
        std::string lightType = jsnNode[NodeNames::TYPE];

        std::unique_ptr<SceneNodeI> lightNode = nullptr;
        lightNode = std::make_unique<SceneNodeJson>(jsnNode, nIndex);
        lightNode->AddIdIndexPair(lightId, iIndex);

        if(lightType == NodeNames::LIGHT_TYPE_PRIMITIVE)
            return SceneError(SceneError::PRIM_BACKED_LIGHT_AS_BOUNDARY,
                              std::to_string(lightId));

        LightGroupData data =
        {
            BaseConstants::EMPTY_PRIMITIVE_NAME,
            std::vector<EndpointConstructionData>()
        };
        std::string groupName = MangledNames::LightGroup(lightType.c_str());
        auto& lightData = lightGroupNodes.emplace(groupName, std::move(data)).first->second;
        auto& constructionInfo = lightData.constructionInfo;
        constructionInfo.emplace_back(EndpointConstructionData
                                      {
                                          surfId,
                                          boundaryTransformId,
                                          baseMediumId,
                                          lightId,
                                          std::numeric_limits<uint32_t>::max(),
                                          std::move(lightNode),
                                      });

        bLightId = lightId;
        bLightGroupTypeName = groupName;
    }
    else return SceneError(SceneError::LIGHT_ID_NOT_FOUND, std::to_string(lightId));

    // Finally Load Texture Info for material access
    // Load all textures here (as node).
    // Materials will actually load the textures
    for(const auto& jsn : (*texturesJson))
    {
        std::vector<TextureStruct> texStructs = SceneIO::LoadTexture(jsn);
        for(const auto& t : texStructs)
            textureNodes.emplace(t.texId, t);
    }
    return e;
}

SceneError GPUSceneJson::GenerateMaterialGroups(const MultiGPUMatNodes& matGroupNodes,
                                                const TextureNodeMap& textureNodes,
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
        if((e = logicGenerator.GenerateMaterialGroup(matGroup, *gpu, matTypeName)) != SceneError::OK)
            return e;
        if((e = matGroup->InitializeGroup(matNodes, textureNodes,
                                         mediumIdMappings, time, parentPath)) != SceneError::OK)
            return e;
        materials.emplace(std::make_pair(matTypeName, gpu), std::move(matGroup));
    }
    return e;
}

SceneError GPUSceneJson::GenerateWorkBatches(MaterialKeyListing& allMatKeys,
                                             // I-O
                                             uint32_t& currentBatchCount,
                                             // Input
                                             const MultiGPUWorkBatches& materialBatches)
{
    SceneError e = SceneError::OK;
    // First do materials
    for(const auto& requiredMat : materialBatches)
    {
        currentBatchCount++;
        if(currentBatchCount >= (1 << HitKey::BatchBits))
            return SceneError(SceneError::TOO_MANY_MATERIAL_GROUPS,
                              std::to_string(currentBatchCount));

        // Generate Keys
        const CudaGPU* gpu = requiredMat.first.second;
        const std::string& matTName = requiredMat.second.matType;
        const std::string& primTName = requiredMat.second.primType;
        // Find Interfaces
        // and generate work info
        GPUPrimitiveGroupI* pGroup = primitives.at(primTName).get();
        GPUMaterialGroupI* mGroup = materials.at(std::make_pair(matTName, gpu)).get();
        workInfo.emplace_back(currentBatchCount, pGroup, mGroup);

        // Generate Keys
        // Find inner ids of those materials
        // And combine a key
        const GPUMaterialGroupI& matGroup = *mGroup;
        for(const auto& matId : requiredMat.second.matIds)
        {
            uint32_t innerId = matGroup.InnerId(matId);
            HitKey key = HitKey::CombinedKey(currentBatchCount, innerId);
            allMatKeys.emplace(std::make_pair(primTName, matId), key);

            maxMatIds = Vector2i::Max(maxMatIds, Vector2i(currentBatchCount, innerId));
        }
    }
    return e;
}

SceneError GPUSceneJson::GeneratePrimitiveGroups(const PrimitiveNodeList& primGroupNodes,
                                                 const TextureNodeMap& textureNodes,
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
        if((e = logicGenerator.GeneratePrimitiveGroup(pg, primTypeName)) != SceneError::OK)
            return e;
        if((e = pg->InitializeGroup(primNodes, time, surfaceLoaderGenerator,
                                   textureNodes, parentPath)) != SceneError::OK)
            return e;

        ExpandHitStructSize(*pg.get());
        primitives.emplace(primTypeName, std::move(pg));
    }
    return e;
}

SceneError GPUSceneJson::GenerateAccelerators(std::map<uint32_t, HitKey>& accHitKeyList,
                                              //
                                              const AcceleratorBatchList& acceleratorBatchList,

                                              const MaterialKeyListing& workKeyList,
                                              const BoundaryMaterialKeyListing& boundaryWorkKeyList,
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
            return SceneError(SceneError::TOO_MANY_ACCELERATOR_GROUPS,
                              std::to_string(accelBatch));

        const uint32_t accelId = accelBatch;

        const std::string& accelGroupName = accelGroupBatch.second.accelType;
        const auto& primTName = accelGroupBatch.second.primType;
        const auto& surfaceList = accelGroupBatch.second.surfaces;
        const auto& lightSurfaceList = accelGroupBatch.second.lightSurfaces;
        const auto& accelNode = accelGroupBatch.second.accelNode;


        std::map<uint32_t, SurfaceDefinition> surfaces;
        for(const auto& s : surfaceList)
        {
            SurfaceDefinition surfaceDef;
            surfaceDef.primIdWorkKeyPairs.fill(IdKeyPair{std::numeric_limits<uint32_t>::max(),
                                                         HitKey::InvalidKey});
            const AccelGroupData::SurfaceDef& surface = s.second;

            // Find Transform Index
            surfaceDef.globalTransformIndex = transformIdMappings.at(surface.transformId);

            // Do not expand the keys for each primitive
            surfaceDef.doKeyExpansion = false;

            // Convert Prim / Material Ids with Work Keys
            for(int i = 0; i < SceneConstants::MaxPrimitivePerSurface; i++)
            {
                const IdPair& p = surface.matPrimIdPairs[i];
                if(p.first == std::numeric_limits<uint32_t>::max())
                    break;

                const auto matLookupKey = std::make_pair(primTName, p.first);
                HitKey workKey = workKeyList.at(matLookupKey);
                surfaceDef.primIdWorkKeyPairs[i] = IdKeyPair{p.second, workKey};
            }
            surfaces.emplace(s.first, surfaceDef);
        }
        for(const auto& s : lightSurfaceList)
        {
            SurfaceDefinition surfaceDef;
            surfaceDef.primIdWorkKeyPairs.fill(IdKeyPair{std::numeric_limits<uint32_t>::max(),
                                                         HitKey::InvalidKey});
            const AccelGroupData::LSurfaceDef& lightSurface = s.second;

            uint32_t transformId = lightSurface.transformId;
            uint32_t primitiveId = lightSurface.primId;
            uint32_t lightId = lightSurface.lightId;

            // Find Transform Index
            surfaceDef.globalTransformIndex = transformIdMappings.at(lightSurface.transformId);

            // For each individual primitive there should be different key
            // boundary work key list holds the first key of the primitive batch
            surfaceDef.doKeyExpansion = true;

            // Find Boundary Material Key for this transform/prim
            std::string lightTName = MangledNames::LightGroup(primTName.c_str());
            const auto matLookupKey = std::make_tuple(lightTName, lightId, transformId);
            HitKey workKey = boundaryWorkKeyList.at(matLookupKey);
            surfaceDef.primIdWorkKeyPairs[0] = IdKeyPair{primitiveId, workKey};

            surfaces.emplace(s.first, surfaceDef);
        }

        // Fetch Primitive
        GPUPrimitiveGroupI* pGroup = primitives.at(primTName).get();

        // Group Generation
        GPUAccelGPtr aGroup = GPUAccelGPtr(nullptr, nullptr);
        if((e = logicGenerator.GenerateAcceleratorGroup(aGroup, *pGroup, accelGroupName)) != SceneError::OK)
            return e;
        if((e = aGroup->InitializeGroup(accelNode,
                                        surfaces,
                                        time)) != SceneError::OK)
            return e;

        // Batch Generation
        accelMap.emplace(accelId, aGroup.get());

        // Now Keys
        // Generate Accelerator Keys...
        const GPUAcceleratorGroupI& accGroup = *aGroup;
        for(const auto& surf : accelGroupBatch.second.surfaces)
        {
            const uint32_t surfId = surf.first;
            uint32_t innerId = accGroup.InnerId(surfId);
            HitKey key = HitKey::CombinedKey(accelId, innerId);
            accHitKeyList.emplace(surfId, key);

            maxAccelIds = Vector2i::Max(maxAccelIds, Vector2i(accelId, innerId));

            if(innerId >= (1 << HitKey::IdBits))
                return SceneError(SceneError::TOO_MANY_ACCELERATOR_IN_GROUP,
                                  accelGroupName);

            // Attach keys of accelerators
            accHitKeyList.emplace(surfId, key);
        }
        for(const auto& lightSurf : accelGroupBatch.second.lightSurfaces)
        {
            const uint32_t surfId = lightSurf.first;
            uint32_t innerId = accGroup.InnerId(surfId);
            HitKey key = HitKey::CombinedKey(accelId, innerId);
            accHitKeyList.emplace(surfId, key);

            maxAccelIds = Vector2i::Max(maxAccelIds, Vector2i(accelId, innerId));

            if(innerId >= (1 << HitKey::IdBits))
                return SceneError(SceneError::TOO_MANY_ACCELERATOR_IN_GROUP,
                                  accelGroupName);

            // Attach keys of accelerators
            accHitKeyList.emplace(surfId, key);
        }

        // Finally emplace it to the list
        accelerators.emplace(accelGroupName, std::move(aGroup));
    }
    return e;
}

SceneError GPUSceneJson::GenerateBaseAccelerator(const std::map<uint32_t, HitKey>& accHitKeyList)
{
    SceneError e = SceneError::OK;

    // Find Base Accelerator Type and generate
    const nlohmann::json* baseAccel = nullptr;
    if(!FindNode(baseAccel, NodeNames::BASE_ACCELERATOR))
        return SceneError::BASE_ACCELERATOR_NODE_NOT_FOUND;
    const std::string baseAccelType = (std::as_const(loadFlags)[SceneLoadFlagType::FORCE_OPTIX_ACCELS])
                                        ? SceneConstants::OptiXAcceleratorTypeName
                                        : std::string((*baseAccel)[NodeNames::TYPE]);
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
                                            uint32_t& boundaryTIndex,
                                            const TransformNodeList& transformList,
                                            uint32_t boundaryTransformId,
                                            double time)
{
    // Generate Transform Groups
    SceneError e = SceneError::OK;
    uint32_t linearIndex = 0;
    //bool hadIdentityTransform = false;
    for(const auto& transformGroup : transformList)
    {
        std::string transTypeName = transformGroup.first;
        const auto& transNodes = transformGroup.second;
        //
        CPUTransformGPtr tg = CPUTransformGPtr(nullptr, nullptr);
        if((e = logicGenerator.GenerateTransformGroup(tg, transTypeName)) != SceneError::OK)
            return e;
        if((e = tg->InitializeGroup(transNodes, time, parentPath)) != SceneError::OK)
            return e;
        transforms.emplace(transTypeName, std::move(tg));

        for(const auto& node : transNodes)
        for(const auto& idPair : node->Ids())
        {
            uint32_t sceneTransId = idPair.first;
            transformIdMappings.emplace(sceneTransId, linearIndex);

            // Set Identity Transform Index
            if(transTypeName == std::string(NodeNames::TRANSFORM_IDENTITY))
                identityTIndex = linearIndex;
            // Set Boundary Transform Index
            if(sceneTransId == boundaryTransformId)
                boundaryTIndex = linearIndex;

            linearIndex++;
        }
    }
    return e;
}

SceneError GPUSceneJson::GenerateMediums(std::map<uint32_t, uint32_t>& mediumIdMappings,
                                         const MediumNodeList& mediumList,
                                         double time)
{
    // Find the base medium and tag its index
    const nlohmann::json* baseMediumNode = nullptr;
    if(!FindNode(baseMediumNode, NodeNames::BASE_MEDIUM))
        return SceneError::BASE_MEDIUM_NODE_NOT_FOUND;
    uint32_t baseMediumId = SceneIO::LoadNumber<uint32_t>(*baseMediumNode, time);

    // Generate Medium Groups
    uint16_t linearIndex = 0;
    bool baseMediumFound = false;
    SceneError e = SceneError::OK;
    for(const auto& mediumGroup : mediumList)
    {
        std::string mediumTypeName = mediumGroup.first;
        const auto& mediumNodes = mediumGroup.second;
        //
        CPUMediumGPtr mg = CPUMediumGPtr(nullptr, nullptr);
        if((e = logicGenerator.GenerateMediumGroup(mg, mediumTypeName)) != SceneError::OK)
            return e;
        if((e = mg->InitializeGroup(mediumNodes, time, parentPath)) != SceneError::OK)
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
        return SceneError(SceneError::MEDIUM_ID_NOT_FOUND,
                          std::to_string(baseMediumId));
    return e;
}

SceneError GPUSceneJson::GenerateCameras(BoundaryMaterialKeyListing& boundaryWorkKeyList,
                                         // I-O
                                         uint32_t& currentBatchCount,
                                         // Input
                                         const CameraNodeList& camGroupList,
                                         const TextureNodeMap& textureNodes,
                                         const std::map<uint32_t, uint32_t>& transformIdMappings,
                                         const std::map<uint32_t, uint32_t>& mediumIdMappings,
                                         double time)
{
    // Generate a special Camera Material Id
    SceneError e = SceneError::OK;
    for(const auto& camGroup : camGroupList)
    {
        currentBatchCount++;
        if(currentBatchCount >= (1 << HitKey::BatchBits))
            return SceneError(SceneError::TOO_MANY_MATERIAL_GROUPS,
                              std::to_string(currentBatchCount));

        const std::string& camTypeName = camGroup.first;
        const auto& camNodes = camGroup.second;
        // Find Empty Primitive
        GPUPrimitiveGroupI* primGroup = nullptr;
        primGroup = primitives.at(BaseConstants::EMPTY_PRIMITIVE_NAME).get();

        CPUCameraGPtr cg = CPUCameraGPtr(nullptr, nullptr);
        if((e = logicGenerator.GenerateCameraGroup(cg, primGroup, camTypeName)) != SceneError::OK)
            return e;
        if((e = cg->InitializeGroup(camNodes,
                                    textureNodes,
                                    mediumIdMappings,
                                    transformIdMappings,
                                    currentBatchCount,
                                    time,
                                    parentPath)) != SceneError::OK)
            return e;

        int i = 0;
        const auto& packedHitKeys = cg->PackedHitKeys();
        assert(packedHitKeys.size() == camNodes.size());
        for(const auto& cInfo : camNodes)
        {
            auto key = std::make_tuple(camTypeName, cInfo.endpointId, cInfo.transformId);
            auto [it, created] = boundaryWorkKeyList.emplace(key, packedHitKeys[i]);

            if(!created) return SceneError(SceneError::OVERLAPPING_CAMERA_FOUND,
                                           "CameraId: " + std::to_string(cInfo.endpointId) + ", " +
                                           "TransformId: " + std::to_string(cInfo.transformId));
            i++;
        }

        maxMatIds = Vector2i::Max(maxMatIds, Vector2i(currentBatchCount, cg->MaxInnerId()));
        boundaryWorkInfo.emplace_back(currentBatchCount, EndpointType::CAMERA, cg.get());
        cameras.emplace(camTypeName, std::move(cg));
    }

    return SceneError::OK;
}

SceneError GPUSceneJson::GenerateLights(BoundaryMaterialKeyListing& boundaryWorkKeyList,
                                        // I-O
                                        uint32_t& currentBatchCount,
                                        // Input
                                        const LightNodeList& lightGroupList,
                                        const TextureNodeMap& textureNodes,
                                        const std::map<uint32_t, uint32_t>& transformIdMappings,
                                        const std::map<uint32_t, uint32_t>& mediumIdMappings,
                                        double time)
{
    SceneError e = SceneError::OK;
    for(const auto& lightGroup : lightGroupList)
    {
        currentBatchCount++;
        if(currentBatchCount >= (1 << HitKey::BatchBits))
            return SceneError(SceneError::TOO_MANY_MATERIAL_GROUPS,
                              std::to_string(currentBatchCount));

        const std::string& lightTypeName = lightGroup.first;
        const std::string& primTypeName = lightGroup.second.primTypeName;
        //bool isPrimLight = lightGroup.second.IsPrimitiveLight();
        const auto& lightNodes = lightGroup.second.constructionInfo;

        // Find Primitive
        GPUPrimitiveGroupI* primGroup = nullptr;
        primGroup = primitives.at(primTypeName).get();

        CPULightGPtr lg = CPULightGPtr(nullptr, nullptr);
        if((e = logicGenerator.GenerateLightGroup(lg, cudaSystem.BestGPU(),
                                                  primGroup, lightTypeName)) != SceneError::OK)
            return e;

        if((e = lg->InitializeGroup(lightNodes,
                                    textureNodes,
                                    mediumIdMappings,
                                    transformIdMappings,
                                    currentBatchCount,
                                    time,
                                    parentPath)) != SceneError::OK)
            return e;

        int i = 0;
        const auto& packedHitKeys = lg->PackedHitKeys();
        assert(packedHitKeys.size() == lightGroup.second.constructionInfo.size());
        for(const auto& lInfo : lightGroup.second.constructionInfo)
        {
            auto key = std::make_tuple(lightTypeName, lInfo.endpointId, lInfo.transformId);
            auto [it, created] = boundaryWorkKeyList.emplace(key, packedHitKeys[i]);

            if(!created) return SceneError(SceneError::OVERLAPPING_LIGHT_FOUND,
                                           "LightId: " + std::to_string(lInfo.endpointId) + ", " +
                                           "TransformId: " + std::to_string(lInfo.transformId));

            i++;
        }

        boundaryWorkInfo.emplace_back(currentBatchCount, EndpointType::LIGHT, lg.get());

        maxMatIds = Vector2i::Max(maxMatIds, Vector2i(currentBatchCount, lg->MaxInnerId()));
        lights.emplace(lightTypeName, std::move(lg));
    }
    return SceneError::OK;
}

SceneError GPUSceneJson::FindBoundaryLight(const BoundaryMaterialKeyListing& bMatKeys,
                                           const std::string& bLightGroupTypeName,
                                           uint32_t bLightId, uint32_t bTransformId)
{
    SceneError e = SceneError::OK;
    // From that node find equivalent material,
    auto tripletKey = std::make_tuple(bLightGroupTypeName, bLightId,
                                      bTransformId);
    auto loc = bMatKeys.find(tripletKey);
    if(loc == bMatKeys.end())
        return SceneError(SceneError::LIGHT_ID_NOT_FOUND,
                          std::to_string(bLightId));

    baseBoundaryMatKey = loc->second;
    return e;
}

SceneError GPUSceneJson::LoadAll(double time)
{
    Utility::CPUTimer timer;
    timer.Start();

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
    TextureNodeMap textureNodes;
    uint32_t boundaryTransformId;
    std::string bLightGroupTypeName;
    uint32_t bLightId;
    // Parse Json and find necessary nodes
    if((e = GenerateConstructionData(primGroupNodes,
                                     mediumGroupNodes,
                                     transformGroupNodes,
                                     boundaryTransformId,
                                     bLightGroupTypeName,
                                     bLightId,
                                     matGroupNodes,
                                     workListings,
                                     accelListings,
                                     camListings,
                                     lightListings,
                                     textureNodes,
                                     time)) != SceneError::OK)
        return e;

    // Transforms
    std::map<uint32_t, uint32_t> transformIdMappings;
    if((e = GenerateTransforms(transformIdMappings,
                               identityTransformIndex,
                               boundaryTransformIndex,
                               transformGroupNodes,
                               boundaryTransformId,
                               time)) != SceneError::OK)
        return e;

    // Mediums
    std::map<uint32_t, uint32_t> mediumIdMappings;
    if((e = GenerateMediums(mediumIdMappings,
                            mediumGroupNodes,
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

    // Using those constructs generate
    // Primitive Groups
    if((e = GeneratePrimitiveGroups(primGroupNodes, textureNodes,
                                    time)) != SceneError::OK)
        return e;
    // Material Groups
    if((e = GenerateMaterialGroups(multiGPUMatNodes, textureNodes,
                                   mediumIdMappings, time)) != SceneError::OK)
        return e;

    // Work Batches
    uint32_t batchId = NullBatchId;
    MaterialKeyListing allMaterialKeys;
    if((e = GenerateWorkBatches(allMaterialKeys,
                                batchId,
                                multiGPUWorkBatches)) != SceneError::OK)
        return e;

    // Generate Endpoints
    // in meantime generate allBoundaryWorkKeysAswell
    BoundaryMaterialKeyListing allBoundaryMaterialKeys;
    // Cameras
    if((e = GenerateCameras(allBoundaryMaterialKeys,
                            batchId,
                            camListings,
                            textureNodes,
                            transformIdMappings,
                            mediumIdMappings,
                            time)) != SceneError::OK)
        return e;
    // Lights
    if((e = GenerateLights(allBoundaryMaterialKeys,
                           batchId,
                           lightListings,
                           textureNodes,
                           transformIdMappings,
                           mediumIdMappings,
                           time)) != SceneError::OK)
        return e;

    // Accelerators
    std::map<uint32_t, HitKey> accHitKeyList;
    if((e = GenerateAccelerators(accHitKeyList, accelListings,
                                 allMaterialKeys,
                                 allBoundaryMaterialKeys,
                                 transformIdMappings,
                                 time)) != SceneError::OK)
        return e;
    // Base Accelerator
    if((e = GenerateBaseAccelerator(accHitKeyList)) != SceneError::OK)
        return e;
    // Boundary Light
    if((e = FindBoundaryLight(allBoundaryMaterialKeys,
                              bLightGroupTypeName,
                              bLightId, boundaryTransformId)) != SceneError::OK)
       return e;

    // MaxIds are generated but those are inclusive
    // Make them exclusive
    maxAccelIds += Vector2i(1);
    maxMatIds += Vector2i(1);

    timer.Stop();
    METU_LOG("Scene {:s} loaded in {:f} seconds.",
             Utility::CopyU8ToString(fileName),
             timer.Elapsed<CPUTimeSeconds>());

    // Everything is generated!
    return SceneError::OK;
}

SceneError GPUSceneJson::ChangeAll(double)
{
    // TODO:
    return SceneError(SceneError::TYPE_MISMATCH,
                      "\"GPUSceneJson\" not yet implemented change time");
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
    Utility::CPUTimer t;
    t.Start();
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
        if(e.what() != nullptr)
            METU_ERROR_LOG("{:s}", e.what());
        return e;
    }
    catch(nlohmann::json::parse_error const& e)
    {
        METU_ERROR_LOG("{:s}", e.what());
        return SceneError::JSON_FILE_PARSE_ERROR;
    }
    t.Stop();
    loadTime = t.Elapsed<CPUTimeSeconds>();
    return e;
}

SceneError GPUSceneJson::ChangeTime(double time)
{
    Utility::CPUTimer t;
    t.Start();
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
    t.Stop();
    lastUpdateTime = t.Elapsed<CPUTimeSeconds>();
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

double GPUSceneJson::MaxSceneTime() const
{
    // TODO: change this when animation is fully implemented
    return 0.0;
}

uint32_t GPUSceneJson::CameraCount() const
{
    return cameraCount;
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

uint16_t GPUSceneJson::BaseMediumIndex() const
{
    return baseMediumIndex;
}

uint32_t GPUSceneJson::IdentityTransformIndex() const
{
    return identityTransformIndex;
}

uint32_t GPUSceneJson::BoundaryTransformIndex() const
{
    return boundaryTransformIndex;
}

const WorkBatchCreationInfo& GPUSceneJson::WorkBatchInfo() const
{
    return workInfo;
}

const BoundaryWorkBatchCreationInfo& GPUSceneJson::BoundarWorkBatchInfo() const
{
    return boundaryWorkInfo;
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

SceneAnalyticData GPUSceneJson::AnalyticData() const
{
    return SceneAnalyticData
    {
        Utility::PathFile(Utility::CopyU8ToString(fileName)),
        loadTime,
        lastUpdateTime,
        {
            static_cast<uint32_t>(MaterialGroups().size()),
            static_cast<uint32_t>(PrimitiveGroups().size()),
            static_cast<uint32_t>(Lights().size()),
            static_cast<uint32_t>(Cameras().size()),
            static_cast<uint32_t>(AcceleratorGroups().size()),
            static_cast<uint32_t>(Transforms().size()),
            static_cast<uint32_t>(Mediums().size())
        },
        MaxAccelIds(),
        MaxMatIds(),
    };
}
