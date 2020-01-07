#pragma once

#include "GPUEventEstimatorI.h"
#include "EstimatorFunctions.cuh"
#include "EstimatorStructs.h"
#include "GPUPrimitiveI.h"

#include "RayLib/SceneNodeNames.h"

#include <vector>

template <class EstimatorD>
class GPUEventEstimatorP
{
    friend struct EstimatorDataAccessor;

    protected:
        EstimatorD dData = EstimatorD{};
};

template<class EstimatorD, 
         EstimateEventFunc<EstimatorD> EstF,
         TerminateEventFunc TermF>
class GPUEventEstimator
    : public GPUEventEstimatorI
    , public GPUEventEstimatorP<EstimatorD>
{
    public:
        // Type Definitions
        using EstimatorData                     = typename EstimatorD;
        // Function Definitions
        static constexpr auto EstimatorFunc     = EstF;
        static constexpr auto TerminatorFunc    = TermF;

    private:
    protected:
        std::vector<EstimatorInfo>              lightInfo;

    public:
        // Constructors & Destructor
                                GPUEventEstimator() = default;
        virtual                 ~GPUEventEstimator() = default;

        virtual SceneError      Initialize(const NodeListing& lightList,
                                           // Material Keys
                                           const MaterialKeyListing& hitKeys,
                                           const std::map<uint32_t, const GPUPrimitiveGroupI*>&,
                                           double time) override;
};

struct EstimatorDataAccessor
{
    // Data fetch function of the primitive
    // This struct should contain all necessary data required for kernel calls
    // related to this primitive
    // I dont know any design pattern for converting from static polymorphism
    // to dynamic one. This is my solution (it is quite werid)
    template <class EventEstimatorS>
    static typename EventEstimatorS::EstimatorData Data(const EventEstimatorS& est)
    {
        using E = typename EventEstimatorS::EstimatorData;
        return static_cast<const GPUEventEstimatorP<E>&>(est).dData;
    }
};

template<class EstimatorD, EstimateEventFunc<EstimatorD> EstF, TerminateEventFunc TermF>
SceneError GPUEventEstimator<EstimatorD, EstF, TermF>::Initialize(const NodeListing& lightList,
                                                                  // Material Keys
                                                                  const MaterialKeyListing& hitKeys,
                                                                  const std::map<uint32_t, const GPUPrimitiveGroupI*>& prims,
                                                                  double time)
{
    SceneError err = SceneError::OK;
    for(const auto& nodePtr : lightList)
    {
        const SceneNodeI& node = *nodePtr;

        const std::string typeName = node.CommonString(NodeNames::TYPE, time);
        LightType type = LightType::END;
        if((err = LightTypeStringToEnum(type, typeName)) != SceneError::OK)
            return err;
        
        switch(type)
        {
            // Fetch from Node if analytic light
            case LightType::POINT:
            case LightType::DIRECTIONAL:
            case LightType::SPOT:
            case LightType::RECTANGULAR:
            case LightType::TRIANGULAR:
            case LightType::DISK:
            case LightType::SPHERICAL:
            {
                lightInfo.push_back({});
                if((err = FetchLightInfoFromNode(lightInfo.back(), node, type)) != SceneError::OK)
                   return err;
                break;
            }
            case LightType::PRIMITIVE:
            { 
                const UIntList primIds = node.AccessUInt(NodeNames::LIGHT_PRIMITIVE);

                // Fetch all
                for(uint32_t prim : primIds)
                {
                    auto i = prims.end();
                    if((i = prims.find(prim)) == prims.end())
                        continue;
                    // Generate Estimators From Primitive
                    std::vector<EstimatorInfo> info;
                    if((err = i->second->GenerateEstimatorInfo(info, prim)) != SceneError::OK)
                        return err;
                    // Insert
                    lightInfo.insert(lightInfo.end(), info.begin(), info.end());
                }
                break;
            }
            default: return SceneError::UNKNOWN_LIGHT_TYPE;
        }
    }
    return SceneError::OK;
}