#pragma once

#include "GPUEventEstimatorI.h"
#include "EstimatorFunctions.cuh"

enum class LightType
{
    POINT,
    DIRECTIONAL,
    SPOT,
    RECTANGULAR,
    TRIANGULAR,
    DISK,
    SPHERICAL
};

struct EstimatorInfo
{
    LightType   type;

    Vector4     position0X;
    Vector4     position1Y;
    Vector4     position2Z;
};

template <class EstimatorD>
class GPUEventEstimatorP
{
    friend struct EstimatorDataAccessor;

    protected:
        EstimatorD dData = EstimatorD{};
};

template<class EstimatorD, EstimateEventFunc<EstimatorD> EstF>
class GPUEventEstimator
    : public GPUEventEstimatorI
    , public GPUEventEstimatorP
{
    public:
        // Type Definitions
        using EstimatorData = EstimatorD;
        // Function Definitions
        static constexpr auto EstimatorFunc = EstF;

    private:
    protected:
        // Array of Light Data That is not converted to utilized data
        std::vector<LightInfo>    lightData;
        std::vector<HitKey>       lightMaterialKey;

    public:
        // Constructors & Destructor
                            GPUEventEstimator() = default;
        virtual             ~GPUEventEstimator() = default;

        SceneError          Initialize(const NodeListing& lightList,
                                       // Material Keys
                                       const MaterialKeyListing& hitKeys,
                                       const std::map<uint32_t, GPUPrimitiveGroupI>&) override;
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
        return static_cast<const EventEstimatorS<E>&>(est).dData;
    }
};

template<class EstimatorD, EstimateEventFunc<EstimatorD> EstF>
SceneError GPUEventEstimator<EstimatorD, EstF>::Initialize(const NodeListing& lightList,
                                                           // Material Keys
                                                           const MaterialKeyListing& hitKeys,
                                                           const std::map<uint32_t, GPUPrimitiveGroupI>&)
{


    return SceneError::OK;
}