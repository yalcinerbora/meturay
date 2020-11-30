#pragma once

#include <map>
#include <string>
#include <regex>

#include "TracerTypeGenerators.h"

class AcceleratorLogicPoolI
{
    protected:
        std::map<std::string, GPUAccelGroupGen>   acceleratorGroupGenerators;
    public:
        static constexpr const char* DefaultConstructorName = "GenAcceleratorPool";
        static constexpr const char* DefaultDestructorName = "DelAcceleratorPool";

        virtual                     ~AcceleratorLogicPoolI() = default;

        virtual std::map<std::string, GPUAccelGroupGen> AcceleratorGroupGenerators(const std::string regex = ".*") const;
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

    public:
        static constexpr const char* DefaultConstructorName = "GenMaterialPool";
        static constexpr const char* DefaultDestructorName = "DelMaterialPool";

        virtual                     ~MaterialLogicPoolI() = default;

        virtual std::map<std::string, GPUMatGroupGen> MaterialGroupGenerators(const std::string regex = ".*") const;
};

class TracerPoolI
{
    protected:
        std::map<std::string, GPUTracerGen>   tracerGenerators;

    public:
        static constexpr const char* DefaultConstructorName = "GenTracerPool";
        static constexpr const char* DefaultDestructorName = "DelTracerPool";

        virtual                         ~TracerPoolI() = default;

        virtual std::map<std::string, GPUTracerGen> TracerGenerators(const std::string regex = ".*") const;
};

class TransformPoolI
{
    protected:
        std::map<std::string, CPUTransformGen>   transformGenerators;

    public:
        static constexpr const char* DefaultConstructorName = "GenTransformPool";
        static constexpr const char* DefaultDestructorName = "DelTransformPool";

        virtual                         ~TransformPoolI() = default;

        virtual std::map<std::string, CPUTransformGen> TransformGenerators(const std::string regex = ".*") const;
};

class MediumPoolI
{
    protected:
        std::map<std::string, CPUMediumGen>     mediumGenerators;

    public:
        static constexpr const char* DefaultConstructorName = "GenMediumPool";
        static constexpr const char* DefaultDestructorName = "DelMediumPool";

        virtual                                 ~MediumPoolI() = default;

        virtual std::map<std::string, CPUMediumGen> MediumGenerators(const std::string regex = ".*") const;
};

class CameraPoolI
{
    protected:
        std::map<std::string, CPUCameraGen>   cameraGenerators;

    public:
        static constexpr const char* DefaultConstructorName = "GenCameraPool";
        static constexpr const char* DefaultDestructorName = "DelCameraPool";

        virtual                         ~CameraPoolI() = default;

        virtual std::map<std::string, CPUCameraGen> CameraGenerators(const std::string regex = ".*") const;
};

class LightPoolI
{
    protected:
        std::map<std::string, GPULightGroupGen>   lightGenerators;

    public:
        static constexpr const char* DefaultConstructorName = "GenLightPool";
        static constexpr const char* DefaultDestructorName = "DelLightPool";

        virtual                         ~LightPoolI() = default;

        virtual std::map<std::string, GPULightGroupGen> LightGenerators(const std::string regex = ".*") const;
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

inline std::map<std::string, GPUTracerGen> TracerPoolI::TracerGenerators(const std::string regex) const
{
    std::map<std::string, GPUTracerGen> result;
    std::regex regExpression(regex);
    for(const auto& tracerGenerator : tracerGenerators)
    {
        if(std::regex_match(tracerGenerator.first, regExpression))
            result.emplace(tracerGenerator);
    }
    return result;
}

inline std::map<std::string, CPUTransformGen> TransformPoolI::TransformGenerators(const std::string regex) const
{
    std::map<std::string, CPUTransformGen> result;
    std::regex regExpression(regex);
    for(const auto& transformGenerator : transformGenerators)
    {
        if(std::regex_match(transformGenerator.first, regExpression))
            result.emplace(transformGenerator);
    }
    return result;
}

inline std::map<std::string, CPUMediumGen> MediumPoolI::MediumGenerators(const std::string regex) const
{
    std::map<std::string, CPUMediumGen> result;
    std::regex regExpression(regex);
    for(const auto& mediumGenerator : mediumGenerators)
    {
        if(std::regex_match(mediumGenerator.first, regExpression))
            result.emplace(mediumGenerator);
    }
    return result;
}

inline std::map<std::string, CPUCameraGen> CameraPoolI::CameraGenerators(const std::string regex) const
{
    std::map<std::string, CPUCameraGen> result;
    std::regex regExpression(regex);
    for(const auto& cameraGenerator : cameraGenerators)
    {
        if(std::regex_match(cameraGenerator.first, regExpression))
            result.emplace(cameraGenerator);
    }
    return result;
}

inline std::map<std::string, GPULightGroupGen> LightPoolI::LightGenerators(const std::string regex) const
{
    std::map<std::string, GPULightGroupGen> result;
    std::regex regExpression(regex);
    for(const auto& lightGenerator : lightGenerators)
    {
        if(std::regex_match(lightGenerator.first, regExpression))
            result.emplace(lightGenerator);
    }
    return result;
}