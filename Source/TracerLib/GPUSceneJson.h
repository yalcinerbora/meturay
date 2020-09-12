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

using MediumNodeList = std::map<std::string, NodeListing>;
using TransformNodeList = std::map<std::string, NodeListing>;
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
            MEDIUM
        };

    private:

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
        uint32_t                                identityTransformIndex;
        uint32_t                                baseMediumIndex;

        // GPU Memory
        GPUBaseAccelPtr                         baseAccelerator;
        std::map<NameGPUPair, GPUMatGPtr>       materials;
        NamedList<GPUAccelGPtr>                 accelerators;
        NamedList<GPUPrimGPtr>                  primitives;
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
        NamedList<CPUTransformGPtr>             transforms;
        NamedList<CPUMediumGPtr>                mediums;

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
                                                 //
                                                 MaterialNodeList& matGroupNodes,
                                                 WorkBatchList& matBatchListings,
                                                 AcceleratorBatchList& accelBatchListings,
                                                 //
                                                 double time = 0.0);
        SceneError      GeneratePrimitiveGroups(const PrimitiveNodeList&,
                                                double time = 0.0);
        SceneError      GenerateMaterialGroups(const MultiGPUMatNodes&,
                                               const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                               double time = 0.0);
        SceneError      GenerateWorkBatches(MaterialKeyListing&,
                                            const MultiGPUWorkBatches&,
                                            double time = 0.0);
        SceneError      GenerateAccelerators(std::map<uint32_t, HitKey>& accHitKeyList,
                                             //
                                             const AcceleratorBatchList& acceleratorBatchList,
                                             const MaterialKeyListing& matHitKeyList,
                                             //
                                             const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                             double time = 0.0);
        SceneError      GenerateBaseAccelerator(const std::map<uint32_t, HitKey>& accHitKeyList,
                                                double time = 0.0);
        SceneError      GenerateLightInfo(const MaterialKeyListing& materialKeys,
                                          double time);
        SceneError      GenerateTransforms(std::map<uint32_t, uint32_t>& surfaceTransformIds,
                                           uint32_t& identityTransformIndex,
                                           const TransformNodeList& transformList,
                                           double time);

        SceneError      GenerateMediums(std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                        uint32_t& baseMediumIndex,
                                        const MediumNodeList& mediumList,
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

        const NamedList<CPUTransformGPtr>&  Transforms() const override;
        const NamedList<CPUMediumGPtr>&     Mediums() const override;
                
        uint32_t                            BaseMediumIndex() const override;
        uint32_t                            IdentityTransformIndex() const override;

        // Generated Classes of Materials / Accelerators
        // Work Maps
        const WorkBatchCreationInfo&        WorkBatchInfo() const override;
        const AcceleratorBatchMap&          AcceleratorBatchMappings() const override;

        // Allocated Types
        // All of which are allocated on the GPU
        const GPUBaseAccelPtr&                      BaseAccelerator() const override;
        const std::map<NameGPUPair, GPUMatGPtr>&    MaterialGroups() const override;
        const NamedList<GPUAccelGPtr>&              AcceleratorGroups() const override;
        const NamedList<GPUPrimGPtr>&               PrimitiveGroups() const override;
};
