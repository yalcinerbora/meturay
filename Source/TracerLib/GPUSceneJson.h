#pragma once

#include <vector>
#include <nlohmann/json_fwd.hpp>

#include "RayLib/Camera.h"
#include "RayLib/AABB.h"
#include "RayLib/GPUSceneI.h"

#include "DeviceMemory.h"
#include "CudaConstants.h"
#include "ScenePartitionerI.h"

struct SceneError;
class SceneNodeI;
//class ScenePartitionerI;
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

        // Fundamental Allocators
        TracerLogicGeneratorI&                  logicGenerator;
        ScenePartitionerI&                      partitioner;
        const SurfaceLoaderGeneratorI&          surfaceLoaderGenerator;

        // Loaded
        Vector2i                                maxAccelIds;
        Vector2i                                maxMatIds;
        HitKey                                  baseBoundaryMatKey;
        uint32_t                                hitStructSize;

        // GPU Memory
        DeviceMemory                            memory;
        GPUBaseAccelPtr                         baseAccelerator;
        std::map<NameGPUPair, GPUMatGPtr>       materials;
        std::map<std::string, GPUAccelGPtr>     accelerators;
        std::map<std::string, GPUPrimGPtr>      primitives;
        // CPU Memory
        std::vector<CameraPerspective>          cameraMemory;
        WorkBatchCreationInfo                   workInfo;
        AcceleratorBatchMap                     accelMap;

        // File Related
        std::unique_ptr<nlohmann::json>         sceneJson;
        std::u8string                           fileName;
        std::string                             parentPath;
        double                                  currentTime;

        // GPU Pointers
        size_t                                  lightCount;
        size_t                                  cameraCount;
        size_t                                  transformCount;
        GPULightI*                              dLights;
        GPUCameraI*                             dCameras;
        TransformStruct*                        dTransforms;

        // Inners
        // Helper Logic
        SceneError                              OpenFile(const std::u8string& fileName);
        bool                                    FindNode(const nlohmann::json*& node, const char* name);
        static SceneError                       GenIdLookup(IndexLookup&,
                                                            const nlohmann::json& array,
                                                            IdBasedNodeType);
        void                                    ExpandHitStructSize(const GPUPrimitiveGroupI& pg);

        // Private Load Functionality
        SceneError      GenerateConstructionData(// Striped Listings (Striped from unsued nodes)
                                                 PrimitiveNodeList& primGroupNodes,
                                                 //
                                                 MaterialNodeList& matGroupNodes,
                                                 WorkBatchList& matBatchListings,
                                                 AcceleratorBatchList& accelBatchListings,
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
        SceneError      GenerateWorkBatches(MaterialKeyListing&,
                                            const MultiGPUWorkBatches&,
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
        SceneError      GenerateLightInfo(std::vector<LightStruct>&,
                                          const MaterialKeyListing& materialKeys, double time);
        SceneError      FindBoundaryMaterial(const MaterialKeyListing& matHitKeyList,
                                             double time = 0.0f);

        SceneError      LoadCommon(const std::vector<LightStruct>&, double time);
        SceneError      LoadLogicRelated(std::vector<LightStruct>&, double time);

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
        GPUSceneJson&                       operator=(const GPUSceneJson&) = delete;
                                            ~GPUSceneJson() = default;

        // Members
        size_t                              UsedGPUMemory() override;
        size_t                              UsedCPUMemory() override;
        //
        SceneError                          LoadScene(double) override;
        SceneError                          ChangeTime(double) override;
        //
        Vector2i                            MaxMatIds() override;
        Vector2i                            MaxAccelIds() override;
        HitKey                              BaseBoundaryMaterial() override;
        // Access GPU
        const GPULightI*                    LightsGPU() const override;
        const GPUCameraI*                   CamerasGPU() const override;
        const TransformStruct*              TransformsGPU() const  override;
        
        // Counts
        size_t                              LightCount() const override;
        size_t                              TransformCount() const override;
        size_t                              CameraCount() const override;

        // Generated Classes of Materials / Accelerators
        // Work Maps
        const WorkBatchCreationInfo&        WorkBatchInfo() const override;
        const AcceleratorBatchMap&          AcceleratorBatchMappings() const override;

        // Allocated Types
        // All of which are allocated on the GPU
        const GPUBaseAccelPtr&                          BaseAccelerator() const override;
        const std::map<NameGPUPair, GPUMatGPtr>&        MaterialGroups() const override;
        const std::map<std::string, GPUAccelGPtr>&      AcceleratorGroups() const override;
        const std::map<std::string, GPUPrimGPtr>&       PrimitiveGroups() const override;
};
