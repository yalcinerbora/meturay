#pragma once

#include <vector>
#include <nlohmann/json_fwd.hpp>

#include "RayLib/AABB.h"
#include "RayLib/GPUSceneI.h"

#include "DeviceMemory.h"
#include "CudaSystem.h"
#include "ScenePartitionerI.h"

struct SceneError;
class SceneNodeI;
class TracerLogicGeneratorI;
class SurfaceLoaderGeneratorI;

using MediumNodeList = std::map<std::string, NodeListing>;
using TransformNodeList = std::map<std::string, NodeListing>;
using PrimitiveNodeList = std::map<std::string, NodeListing>;
using AcceleratorBatchList = std::map<std::string, AccelGroupData>;
using LightNodeList = std::map<std::string, LightGroupData>;
using CameraNodeList = std::map<std::string, EndpointGroupDataList>;
using TextureNodeMap = std::map<uint32_t, TextureStruct>;

class GPUSceneJson : public GPUSceneI
{
    public:
        enum IdBasedNodeType
        {
            ACCELERATOR,
            MATERIAL,
            PRIMITIVE,
            TRANSFORM,
            MEDIUM,
            LIGHT,
            CAMERA
        };

    private:
        // Fundamental Allocators
        TracerLogicGeneratorI&                  logicGenerator;
        ScenePartitionerI&                      partitioner;
        const SurfaceLoaderGeneratorI&          surfaceLoaderGenerator;
        const CudaSystem&                       cudaSystem;
        // Flags
        SceneLoadFlags                          loadFlags;
        //
        double                                  loadTime;
        double                                  lastUpdateTime;

        // Loaded
        Vector2i                                maxAccelIds;
        Vector2i                                maxMatIds;
        HitKey                                  baseBoundaryMatKey;
        uint32_t                                hitStructSize;
        uint32_t                                identityTransformIndex;
        uint32_t                                boundaryTransformIndex;
        uint16_t                                baseMediumIndex;
        uint32_t                                cameraCount;

        // GPU Memory
        GPUBaseAccelPtr                         baseAccelerator;
        std::map<NameGPUPair, GPUMatGPtr>       materials;
        NamedList<GPUAccelGPtr>                 accelerators;
        NamedList<GPUPrimGPtr>                  primitives;
        // Information of the partitioning
        WorkBatchCreationInfo                   workInfo;
        BoundaryWorkBatchCreationInfo           boundaryWorkInfo;
        AcceleratorBatchMap                     accelMap;

        // File Related
        std::unique_ptr<nlohmann::json>         sceneJson;
        std::u8string                           fileName;
        std::string                             parentPath;
        // TODO: update scene based on time not yet implemented dont remove this
        // instead mark it
        [[maybe_unused]] double                 currentTime;

        // CPU Data
        NamedList<CPUTransformGPtr>             transforms;
        NamedList<CPUMediumGPtr>                mediums;
        NamedList<CPULightGPtr>                 lights;
        NamedList<CPUCameraGPtr>                cameras;

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
                                                 MediumNodeList& mediumGroupNodes,
                                                 TransformNodeList& transformGroupNodes,
                                                 uint32_t& boundaryTransformId,
                                                 //
                                                 std::string& bLightGroupTypeName,
                                                 uint32_t& bLightId,
                                                 //
                                                 MaterialNodeList& matGroupNodes,
                                                 WorkBatchList& matBatchListings,
                                                 AcceleratorBatchList& accelBatchListings,
                                                 //
                                                 CameraNodeList& cameraGroupNodes,
                                                 LightNodeList& lightGroupNodes,
                                                 //
                                                 TextureNodeMap& textureNodes,
                                                 //
                                                 double time = 0.0);
        SceneError      GeneratePrimitiveGroups(const PrimitiveNodeList&,
                                                const TextureNodeMap&,
                                                double time = 0.0);
        SceneError      GenerateMaterialGroups(const MultiGPUMatNodes&,
                                               const TextureNodeMap& textureNodes,
                                               const std::map<uint32_t, uint32_t>& mediumIdMappings,
                                               double time = 0.0);
        SceneError      GenerateWorkBatches(MaterialKeyListing&,
                                            // I-O
                                            uint32_t& currentBatchCount,
                                            // Input
                                            const MultiGPUWorkBatches&);
        SceneError      GenerateAccelerators(std::map<uint32_t, HitKey>& accHitKeyList,
                                             //
                                             const AcceleratorBatchList& acceleratorBatchList,
                                             // Material Key Fetch Related
                                             const MaterialKeyListing& workKeyList,
                                             const BoundaryMaterialKeyListing& boundaryWorkKeyList,
                                             // Transform Id to Global Transform Index
                                             const std::map<uint32_t, uint32_t>& transformIdMappings,
                                             double time = 0.0);
        SceneError      GenerateBaseAccelerator(const std::map<uint32_t, HitKey>& accHitKeyList);
        SceneError      GenerateTransforms(std::map<uint32_t, uint32_t>& transformIdMappings,
                                           uint32_t& identityTransformIndex,
                                           uint32_t& boundaryTransformIndex,
                                           const TransformNodeList& transformList,
                                           uint32_t boundaryTransformId,
                                           double time = 0.0);
        SceneError      GenerateMediums(std::map<uint32_t, uint32_t>& mediumIdMappings,
                                        const MediumNodeList& mediumList,
                                        double time = 0.0);
        SceneError      GenerateCameras(BoundaryMaterialKeyListing&,
                                        // I-O
                                        uint32_t& currentBatchCount,
                                        // Input
                                        const CameraNodeList&,
                                        const TextureNodeMap&,
                                        const std::map<uint32_t, uint32_t>& transformIdMappings,
                                        const std::map<uint32_t, uint32_t>& mediumIdMappings,
                                        double time = 0.0);
        SceneError      GenerateLights(BoundaryMaterialKeyListing&,
                                       // I-O
                                       uint32_t& currentBatchCount,
                                       // Input
                                       const LightNodeList&,
                                       const TextureNodeMap&,
                                       const std::map<uint32_t, uint32_t>& transformIdMappings,
                                       const std::map<uint32_t, uint32_t>& mediumIdMappings,
                                       double time = 0.0);

        SceneError      FindBoundaryLight(const BoundaryMaterialKeyListing& matHitKeyList,
                                          const std::string& bLightGroupTypeName,
                                          uint32_t bLightId, uint32_t bTransformId);

        SceneError      LoadAll(double time);
        SceneError      ChangeAll(double time);

    public:
        // Constructors & Destructor
                                            GPUSceneJson(const std::u8string&,
                                                         ScenePartitionerI&,
                                                         TracerLogicGeneratorI&,
                                                         const SurfaceLoaderGeneratorI&,
                                                         const CudaSystem&,
                                                         SceneLoadFlags = {});
                                            GPUSceneJson(const GPUSceneJson&) = delete;
                                            GPUSceneJson(GPUSceneJson&&) = default;
        GPUSceneJson&                       operator=(const GPUSceneJson&) = delete;
                                            ~GPUSceneJson() = default;

        // Members
        size_t                              UsedGPUMemory() const override;
        size_t                              UsedCPUMemory() const override;
        //
        SceneError                          LoadScene(double) override;
        SceneError                          ChangeTime(double) override;
        //
        Vector2i                            MaxMatIds() const override;
        Vector2i                            MaxAccelIds() const override;
        HitKey                              BaseBoundaryMaterial() const override;
        uint32_t                            HitStructUnionSize() const override;
        double                              MaxSceneTime() const override;
        uint32_t                            CameraCount() const override;

        // Access CPU
        const NamedList<CPULightGPtr>&      Lights() const override;
        const NamedList<CPUCameraGPtr>&     Cameras() const override;

        const NamedList<CPUTransformGPtr>&  Transforms() const override;
        const NamedList<CPUMediumGPtr>&     Mediums() const override;

        uint16_t                            BaseMediumIndex() const override;
        uint32_t                            IdentityTransformIndex() const override;
        uint32_t                            BoundaryTransformIndex() const override;

        // Generated Classes of Materials / Accelerators
        // Work Maps
        const WorkBatchCreationInfo&            WorkBatchInfo() const override;
        const BoundaryWorkBatchCreationInfo&    BoundarWorkBatchInfo() const override;
        const AcceleratorBatchMap&              AcceleratorBatchMappings() const override;

        // Allocated Types
        // All of which are allocated on the GPU
        const GPUBaseAccelPtr&                      BaseAccelerator() const override;
        const std::map<NameGPUPair, GPUMatGPtr>&    MaterialGroups() const override;
        const NamedList<GPUAccelGPtr>&              AcceleratorGroups() const override;
        const NamedList<GPUPrimGPtr>&               PrimitiveGroups() const override;

        //
        SceneAnalyticData                           AnalyticData() const override;
};
