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
        static constexpr const size_t           AlignByteCount = 64;

        // Fundamental Allocators
        TracerLogicGeneratorI&                  logicGenerator;
        ScenePartitionerI&                      partitioner;
        const SurfaceLoaderGeneratorI&          surfaceLoaderGenerator;
        const CudaSystem&                       cudaSystem;

        // Loaded
        Vector2i                                maxAccelIds;
        Vector2i                                maxMatIds;
        HitKey                                  baseBoundaryMatKey;
        uint32_t                                hitStructSize;

        // GPU Memory
        DeviceMemory                            gpuMemory;
        GPUBaseAccelPtr                         baseAccelerator;
        std::map<NameGPUPair, GPUMatGPtr>       materials;
        std::map<std::string, GPUAccelGPtr>     accelerators;
        std::map<std::string, GPUPrimGPtr>      primitives;
        // Information of the partitioning
        WorkBatchCreationInfo                   workInfo;
        AcceleratorBatchMap                     accelMap;

        // File Related
        std::unique_ptr<nlohmann::json>         sceneJson;
        std::u8string                           fileName;
        std::string                             parentPath;
        double                                  currentTime;

        // CPU Data
        std::vector<CPUCamera>                  cameras;
        std::vector<CPULight>                   lights;
        // GPU Pointers
        GPUTransform*                           dTransforms;
        GPUMedium*                              dMediums;
        size_t                                  transformCount;
        size_t                                  mediumCount;

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
        SceneError      GenerateLightInfo(const MaterialKeyListing& materialKeys,
                                          double time);
        SceneError      FindBoundaryMaterial(const MaterialKeyListing& matHitKeyList,
                                             double time = 0.0f);

        SceneError      LoadCommon(double time);
        SceneError      LoadLogicRelated(double time);

        SceneError      ChangeCommon(double time);
        SceneError      ChangeLogicRelated(double time);

    public:
        // Constructors & Destructor
                                            GPUSceneJson(const std::u8string&,
                                                         ScenePartitionerI&,
                                                         TracerLogicGeneratorI&,
                                                         const SurfaceLoaderGeneratorI&,
                                                         const CudaSystem&);
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
        // Access CPU
        const std::vector<CPULight>&        LightsCPU() const override;
        const std::vector<CPUCamera>&       CamerasCPU() const override;
        // Access GPU
        const GPUTransform*                 TransformsGPU() const override;        
        const GPUMedium*                    MediumsGPU() const override;
        // Counts
        size_t                              TransformCount() const override;
        size_t                              MediumCount() const override;
        
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
