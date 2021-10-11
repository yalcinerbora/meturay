//#pragma once
//
//#include "GPULightI.h"
//#include "GPUTransformI.h"
//#include "DeviceMemory.h"
//#include "TypeTraits.h"
//
//class GPULightSpot : public GPULightI
//{
//    private:
//        Vector3             position;
//        float               cosMin;
//        Vector3             direction;
//        float               cosMax;
//
//    protected:
//    public:
//        // Constructors & Destructor
//        __device__              GPULightSpot(// Per Light Data
//                                             const Vector3& position,
//                                             const Vector3& direction,
//                                             const Vector2& aperture,
//                                             const GPUTransformI& gTransform,
//                                             // Endpoint Related Data
//                                             HitKey k, uint16_t mediumIndex);
//                                ~GPULightSpot() = default;
//        // Interface
//        __device__ void         Sample(// Output
//                                       float& distance,
//                                       Vector3& direction,
//                                       float& pdf,
//                                       // Input
//                                       const Vector3& worldLoc,
//                                       // I-O
//                                       RandomGPU&) const override;
//
//        __device__ void         GenerateRay(// Output
//                                            RayReg&,
//                                            // Input
//                                            const Vector2i& sampleId,
//                                            const Vector2i& sampleMax,
//                                            // I-O
//                                            RandomGPU&,
//                                            // Options
//                                            bool antiAliasOn = true) const override;
//
//        __device__ float        Pdf(const Vector3& direction,
//                                    const Vector3& position) const override;
//
//        __device__ bool         CanBeSampled() const override;
//
//        __device__ PrimitiveId  PrimitiveIndex() const override;
//};
//
//class CPULightGroupSpot : public CPULightGroupI
//{
//    public:
//        static constexpr const char*    TypeName(){return "Spot"; }
//
//        static constexpr const char*    NAME_POSITION = "position";
//        static constexpr const char*    NAME_DIRECTION = "direction";
//        static constexpr const char*    NAME_CONE_APERTURE = "aperture";
//
//    private:
//        DeviceMemory                    memory;
//        //
//        std::vector<Vector3f>           hPositions;
//        std::vector<Vector3f>           hDirections;
//        std::vector<Vector2f>           hCosines;
//
//        std::vector<HitKey>             hHitKeys;
//        std::vector<uint16_t>           hMediumIds;
//        std::vector<TransformId>        hTransformIds;
//
//        // Allocations of the GPU Class
//        const GPULightSpot*             dGPULights;
//        // GPU pointers to those allocated classes on the CPU
//        GPULightList				    gpuLightList;
//        uint32_t                        lightCount;
//
//    protected:
//    public:
//        // Cosntructors & Destructor
//                                    CPULightGroupSpot(const GPUPrimitiveGroupI*);
//                                    ~CPULightGroupSpot() = default;
//
//        const char*				    Type() const override;
//		const GPULightList&		    GPULights() const override;
//		SceneError				    InitializeGroup(const LightGroupDataList& lightNodes,
//                                                    const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
//                                                    const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
//                                                    const MaterialKeyListing& allMaterialKeys,
//								    				double time,
//								    				const std::string& scenePath) override;
//		SceneError				    ChangeTime(const NodeListing& lightNodes, double time,
//								    		   const std::string& scenePath) override;
//		TracerError				    ConstructLights(const CudaSystem&,
//                                                    const GPUTransformI**,
//                                                    const KeyMaterialMap&) override;
//		uint32_t				    LightCount() const override;
//
//		size_t					    UsedGPUMemory() const override;
//		size_t					    UsedCPUMemory() const override;
//};
//
//__device__
//inline GPULightSpot::GPULightSpot(// Per Light Data
//                                  const Vector3& position,
//                                  const Vector3& direction,
//                                  const Vector2& aperture,
//                                  const GPUTransformI& gTransform,
//                                  // Endpoint Related Data
//                                  HitKey k, uint16_t mediumIndex)
//    : GPULightI(k, mediumIndex)
//    , position(gTransform.LocalToWorld(position))
//    , direction(gTransform.LocalToWorld(direction, true))
//    , cosMin(aperture[0])
//    , cosMax(aperture[1])
//{}
//
//__device__
//inline void GPULightSpot::Sample(// Output
//                                 float& distance,
//                                 Vector3& dir,
//                                 float& pdf,
//                                 // Input
//                                 const Vector3& worldLoc,
//                                 // I-O
//                                 RandomGPU&) const
//{
//    dir = -direction;
//    distance = (position - worldLoc).Length();
//
//    // Fake pdf to incorporate square faloff
//    pdf = (distance * distance);
//}
//
//__device__
//inline void GPULightSpot::GenerateRay(// Output
//                                      RayReg&,
//                                      // Input
//                                      const Vector2i& sampleId,
//                                      const Vector2i& sampleMax,
//                                      // I-O
//                                      RandomGPU&,
//                                      // Options
//                                      bool antiAliasOn) const
//{
//    // TODO: Implement
//}
//
//__device__
//inline float GPULightSpot::Pdf(const Vector3& worldDir,
//                               const Vector3& worldPos) const
//{
//    return 0.0f;
//}
//
//__device__
//inline bool GPULightSpot::CanBeSampled() const
//{
//    return false;
//}
//
//__device__
//inline PrimitiveId GPULightSpot::PrimitiveIndex() const
//{
//    return INVALID_PRIMITIVE_ID;
//}
//
//inline CPULightGroupSpot::CPULightGroupSpot(const GPUPrimitiveGroupI*)
//    : lightCount(0)
//    , dGPULights(nullptr)
//{}
//
//inline const char* CPULightGroupSpot::Type() const
//{
//    return TypeName();
//}
//
//inline const GPULightList& CPULightGroupSpot::GPULights() const
//{
//    return gpuLightList;
//}
//
//inline uint32_t CPULightGroupSpot::LightCount() const
//{
//    return lightCount;
//}
//
//inline size_t CPULightGroupSpot::UsedGPUMemory() const
//{
//    return memory.Size();
//}
//
//inline size_t CPULightGroupSpot::UsedCPUMemory() const
//{
//    size_t totalSize = (hHitKeys.size() * sizeof(HitKey) +
//                        hMediumIds.size() * sizeof(uint16_t) +
//                        hTransformIds.size() * sizeof(TransformId) +
//                        hPositions.size() * sizeof(Vector3f) +
//                        hDirections.size() * sizeof(Vector3f) +
//                        hCosines.size() * sizeof(Vector2f));
//
//    return totalSize;
//}
//
//static_assert(IsTracerClass<CPULightGroupSpot>::value,
//              "CPULightGroupSpot is not a tracer class");