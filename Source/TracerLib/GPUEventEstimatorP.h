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
        std::vector<HitKey>                     lightMaterialInfo;

    public:
        // Constructors & Destructor
                                GPUEventEstimator() = default;
        virtual                 ~GPUEventEstimator() = default;

        virtual SceneError      Initialize(const NodeListing& lightList,
                                           // Material Keys
                                           const MaterialKeyListing& hitKeys,
                                           const std::vector<const GPUPrimitiveGroupI*>&,
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
                                                                  const MaterialKeyListing& materialKeys,
                                                                  const std::vector<const GPUPrimitiveGroupI*>& prims,
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
                
        std::vector<EstimatorInfo> newInfo;
        if((err = FetchLightInfoFromNode(newInfo, node, materialKeys, type, time)) != SceneError::OK)
            return err;

        if(type == LightType::PRIMITIVE)
        { 
            const UIntList primIds = node.AccessUInt(NodeNames::LIGHT_PRIMITIVE);
            const UIntList matIds = node.AccessUInt(NodeNames::LIGHT_MATERIAL);
            assert(primIds.size() == matIds.size());

            // Fetch all
            std::vector<EstimatorInfo> primInfo;
            for(size_t i = 0; i < primIds.size(); i++)
            {
                uint32_t primId = primIds[i];
                uint32_t matId = matIds[i];

                const auto FindPrimFunc = [primId](const GPUPrimitiveGroupI* pGroup) -> bool
                {
                    return pGroup->HasPrimitive(primId);
                };

                auto it = prims.end();
                if((it = std::find_if(prims.begin(), prims.end(), FindPrimFunc)) == prims.end())
                    return SceneError::LIGHT_PRIMITIVE_NOT_FOUND;
                // Generate Estimators From Primitive

                const auto matLookup = std::make_pair((*it)->Type(), matId);
                HitKey key = materialKeys.at(matLookup);

                if((err = (*it)->GenerateEstimatorInfo(primInfo, key, primId)) != SceneError::OK)
                    return err;

                // Combine
                for(size_t i = 0; i < newInfo.size(); i++)
                    newInfo[i] = EstimatorInfo::CombinePrimEstimators(primInfo[i], newInfo[i]);
            }
        }
        else lightInfo.insert(lightInfo.end(), newInfo.begin(), newInfo.end());
    }
    return SceneError::OK;
}