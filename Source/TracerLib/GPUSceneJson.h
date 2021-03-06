#pragma once

#include <vector>
#include <nlohmann/json_fwd.hpp>

#include "RayLib/Camera.h"
#include "RayLib/SceneStructs.h"
#include "RayLib/GPUSceneI.h"

#include "DeviceMemory.h"
#include "AcceleratorDeviceFunctions.h"
#include "CudaConstants.h"
#include "ScenePartitionerI.h"

struct SceneError;
class SceneNodeI;
class ScenePartitionerI;
class TracerLogicGeneratorI;
class SurfaceLoaderGeneratorI;

using IndexLookup = std::map<NodeId, std::pair<NodeIndex, InnerIndex>>;

using PrimitiveNodeList = std::map<std::string, NodeListing>;
using AcceleratorBatchList = std::map<std::string, AccelGroupData>;

using LightPrimitives = std::vector<const GPUPrimitiveGroupI*>;

class GPUSceneJson : public GPUSceneI
{
    public:
        enum IdBasedNodeType
        {
            ACCELERATOR,
            MATERIAL,
            PRIMITIVE,
            TRANSFORM,
            SURFACE_DATA
        };

    private:
        static constexpr const size_t           AlignByteCount = 128;

        // Fundamental
        TracerLogicGeneratorI&                  logicGenerator;
        ScenePartitionerI&                      partitioner;
        const SurfaceLoaderGeneratorI&          surfaceLoaderGenerator;

        // Loaded
        Vector2i                                maxAccelIds;
        Vector2i                                maxMatIds;
        HitKey                                  baseBoundaryMatKey;

        // GPU Memory
        DeviceMemory                            memory;
        // CPU Memory
        std::vector<CameraPerspective>          cameraMemory;

        // File Related
        std::unique_ptr<nlohmann::json>         sceneJson;
        std::u8string                           fileName;
        std::string                             parentPath;
        double                                  currentTime;

        // GPU Pointers
        LightStruct*                            dLights;
        TransformStruct*                        dTransforms;

        // Inners
        // Helper Logic
        SceneError                              OpenFile(const std::u8string& fileName);
        bool                                    FindNode(const nlohmann::json*& node, const char* name);
        static SceneError                       GenIdLookup(IndexLookup&,
                                                            const nlohmann::json& array,
                                                            IdBasedNodeType);

        // Private Load Functionality
        SceneError      GenerateConstructionData(// Striped Listings (Striped from unsued nodes)
                                                 PrimitiveNodeList& primGroupNodes,
                                                 //
                                                 MaterialNodeList& matGroupNodes,
                                                 MaterialBatchList& matBatchListings,
                                                 AcceleratorBatchList& accelBatchListings,
                                                 // Estimator Related
                                                 NodeListing& lightList,
                                                 // Base Accelerator required data
                                                 std::map<uint32_t, uint32_t>& surfaceTransformIds,
                                                 // Types
                                                 const std::string& estimatorType,
                                                 const std::string& tracerType,
                                                 //
                                                 double time = 0.0);
        SceneError      GeneratePrimitiveGroups(const PrimitiveNodeList&,
                                                double time = 0.0);
        SceneError      GenerateMaterialGroups(const MultiGPUMatNodes&,
                                               double time = 0.0);
        SceneError      GenerateMaterialBatches(MaterialKeyListing&,
                                                const MultiGPUMatBatches&,
                                                double time = 0.0);
        SceneError      GenerateAccelerators(std::map<uint32_t, AABB3>& accAABBs,
                                             std::map<uint32_t, HitKey>& accHitKeyList,
                                             //
                                             const AcceleratorBatchList& acceleratorBatchList,
                                             const MaterialKeyListing& matHitKeyList,
                                             double time = 0.0);
        SceneError      GenerateBaseAccelerator(const std::map<uint32_t, AABB3>& accAABBs,
                                                const std::map<uint32_t, HitKey>& accHitKeyList,
                                                const std::map<uint32_t, uint32_t>& surfaceTransformIds,
                                                double time = 0.0);
        SceneError      FindBoundaryMaterial(const MaterialKeyListing& matHitKeyList,
                                             double time = 0.0f);

        SceneError      LoadCommon(double time);
        SceneError      LoadLogicRelated(const TracerParameters&, double);

        SceneError      ChangeCommon(double time);
        SceneError      ChangeLogicRelated(double time);

    public:
        // Constructors & Destructor
                                    GPUSceneJson(const std::u8string&,
                                                 ScenePartitionerI&,
                                                 TracerLogicGeneratorI&,
                                                 const SurfaceLoaderGeneratorI&);
                                    GPUSceneJson(const GPUSceneJson&) = delete;
                                    GPUSceneJson(GPUSceneJson&&) = default;
        GPUSceneJson&               operator=(const GPUSceneJson&) = delete;
                                    ~GPUSceneJson() = default;

        // Members
        size_t                      UsedGPUMemory() override;
        size_t                      UsedCPUMemory() override;
        //
        SceneError                  LoadScene(const TracerParameters&, double) override;
        SceneError                  ChangeTime(double) override;
        //
        Vector2i                    MaxMatIds() override;
        Vector2i                    MaxAccelIds() override;
        HitKey                      BaseBoundaryMaterial() override;
        // Access GPU
        const LightStruct*          LightsGPU() const override;
        const TransformStruct*      TransformsGPU() const  override;
        // Access CPU
        const CameraPerspective*    CamerasCPU() const override;
};
