#pragma once

#include <map>
#include <string>
#include <regex>

#include "TracerTypeGenerators.h"

class AcceleratorLogicPoolI
{
    protected:
        std::map<std::string, GPUAccelGroupGen>   acceleratorGroupGenerators;
        std::map<std::string, GPUAccelBatchGen>   acceleratorBatchGenerators;
    public:
        static constexpr const char* DefaultConstructorName = "GenAcceleratorPool";
        static constexpr const char* DefaultDestructorName = "DelAcceleratorPool";

        virtual                     ~AcceleratorLogicPoolI() = default;

        virtual std::map<std::string, GPUAccelGroupGen> AcceleratorGroupGenerators(const std::string regex = ".*") const;
        virtual std::map<std::string, GPUAccelBatchGen> AcceleratorBatchGenerators(const std::string regex = ".*") const;
};

class BaseAcceleratorLogicPoolI
{
    protected:
        std::map<std::string, GPUBaseAccelGen>   baseAcceleratorGroupGenerators;

    public:
        static constexpr const char* DefaultConstructorName = "GenBaseAcceleratorPool";
        static constexpr const char* DefaultDestructorName = "DelBaseAcceleratorPool";

        virtual                     ~BaseAcceleratorLogicPoolI() = default;

        virtual std::map<std::string, GPUBaseAccelGen> BaseAcceleratorGenerators(const std::string regex = ".*") const;
};

class PrimitiveLogicPoolI
{
    protected:
        std::map<std::string, GPUPrimGroupGen>   primitiveGenerators;

    public:
        static constexpr const char* DefaultConstructorName = "GenPrimitivePool";
        static constexpr const char* DefaultDestructorName = "DelPrimitivePool";

        virtual                     ~PrimitiveLogicPoolI() = default;

        virtual std::map<std::string, GPUPrimGroupGen> PrimitiveGenerators(const std::string regex = ".*") const;
};

class MaterialLogicPoolI
{
    protected:
        std::map<std::string, GPUMatGroupGen>   materialGroupGenerators;
        std::map<std::string, GPUMatBatchGen>   materialBatchGenerators;

    public:
        static constexpr const char* DefaultConstructorName = "GenMaterialPool";
        static constexpr const char* DefaultDestructorName = "DelMaterialPool";

        virtual                     ~MaterialLogicPoolI() = default;

        virtual std::map<std::string, GPUMatGroupGen> MaterialGroupGenerators(const std::string regex = ".*") const;
        virtual std::map<std::string, GPUMatBatchGen> MaterialBatchGenerators(const std::string regex = ".*") const;
};

class TracerLogicPoolI
{
    protected:
        std::map<std::string, GPUTracerGen>   tracerLogicGenerators;

    public:
        static constexpr const char* DefaultConstructorName = "GenTracerPoolPool";
        static constexpr const char* DefaultDestructorName = "DelTracerPool";

        virtual                         ~TracerLogicPoolI() = default;

        virtual std::map<std::string, GPUTracerGen> TracerGenerators(const std::string regex = ".*") const;
};

class EstimatorLogicPoolI
{
    protected:
        std::map<std::string, GPUEstimatorGen>  estimatorGenerators;

    public:
        static constexpr const char*    DefaultConstructorName = "GenEstimatorPool";
        static constexpr const char*    DefaultDestructorName = "DelEstimatorPool";

        virtual                         ~EstimatorLogicPoolI() = default;

        virtual std::map<std::string, GPUEstimatorGen> EstimatorGenerators(const std::string regex = ".*") const;

};

inline std::map<std::string, GPUAccelGroupGen> AcceleratorLogicPoolI::AcceleratorGroupGenerators(const std::string regex) const
{
    std::map<std::string, GPUAccelGroupGen> result;
    std::regex regExpression(regex);
    for(const auto& batchGenerator : acceleratorGroupGenerators)
    {
        if(std::regex_match(batchGenerator.first, regExpression))
            result.emplace(batchGenerator);
    }
    return result;
}

inline std::map<std::string, GPUAccelBatchGen> AcceleratorLogicPoolI::AcceleratorBatchGenerators(const std::string regex) const
{
    std::map<std::string, GPUAccelBatchGen> result;
    std::regex regExpression(regex);
    for(const auto& batchGenerator : acceleratorBatchGenerators)
    {
        if(std::regex_match(batchGenerator.first, regExpression))
            result.emplace(batchGenerator);
    }
    return result;
}

inline std::map<std::string, GPUBaseAccelGen> BaseAcceleratorLogicPoolI::BaseAcceleratorGenerators(const std::string regex ) const
{
    std::map<std::string, GPUBaseAccelGen> result;
    std::regex regExpression(regex);
    for(const auto& accelGenerator : baseAcceleratorGroupGenerators)
    {
        if(std::regex_match(accelGenerator.first, regExpression))
            result.emplace(accelGenerator);
    }
    return result;
}

inline std::map<std::string, GPUPrimGroupGen> PrimitiveLogicPoolI::PrimitiveGenerators(const std::string regex) const
{
    std::map<std::string, GPUPrimGroupGen> result;
    std::regex regExpression(regex);
    for(const auto& primGenerator : primitiveGenerators)
    {
        if(std::regex_match(primGenerator.first, regExpression))
            result.emplace(primGenerator);
    }
    return result;
}

inline std::map<std::string, GPUMatGroupGen> MaterialLogicPoolI::MaterialGroupGenerators(const std::string regex) const
{
    std::map<std::string, GPUMatGroupGen> result;
    std::regex regExpression(regex);
    for(const auto& groupGenerator : materialGroupGenerators)
    {
        if(std::regex_match(groupGenerator.first, regExpression))
            result.emplace(groupGenerator);
    }
    return result;
}

inline std::map<std::string, GPUMatBatchGen> MaterialLogicPoolI::MaterialBatchGenerators(const std::string regex) const
{
    std::map<std::string, GPUMatBatchGen> result;
    std::regex regExpression(regex);
    for(const auto& batchGenerator : materialBatchGenerators)
    {
        if(std::regex_match(batchGenerator.first, regExpression))
            result.emplace(batchGenerator);
    }
    return result;
}

inline std::map<std::string, GPUTracerGen> TracerLogicPoolI::TracerGenerators(const std::string regex) const
{
    std::map<std::string, GPUTracerGen> result;
    std::regex regExpression(regex);
    for(const auto& tracerGenerator : tracerLogicGenerators)
    {
        if(std::regex_match(tracerGenerator.first, regExpression))
            result.emplace(tracerGenerator);
    }
    return result;
}

inline std::map<std::string, GPUEstimatorGen> EstimatorLogicPoolI::EstimatorGenerators(const std::string regex) const
{
    std::map<std::string, GPUEstimatorGen> result;
    std::regex regExpression(regex);
    for(const auto& estimatorGenerator : estimatorGenerators)
    {
        if(std::regex_match(estimatorGenerator.first, regExpression))
            result.emplace(estimatorGenerator);
    }
    return result;
}