#pragma once

#include <map>
#include <string>

#include "DefaultTypeGenerators.h"

class AcceleratorLogicPoolI
{
    public:
        static constexpr const char* DefaultConstructorName = "GenAcceleratorPool";
        static constexpr const char* DefaultDestructorName = "DelAcceleratorPool";

        virtual std::map<std::string, GPUAccelGroupGen> AcceleratorGroupGenerators(const std::string& regex = "*") const = 0;
        virtual std::map<std::string, GPUAccelBatchGen> AcceleratorBatchGenerators(const std::string& regex = "*") const = 0;
};

class BaseAcceleratorLogicPoolI
{
    public:
        static constexpr const char* DefaultConstructorName = "GenBaseAcceleratorPool";
        static constexpr const char* DefaultDestructorName = "DelBaseAcceleratorPool";

        virtual std::map<std::string, GPUBaseAccelGen> BaseAcceleratorGenerators(const std::string& regex = "*") const = 0;
};

class PrimitiveLogicPoolI
{
    public:
        static constexpr const char* DefaultConstructorName = "GenPrimitivePool";
        static constexpr const char* DefaultDestructorName = "DelPrimitivePool";

        virtual std::map<std::string, GPUPrimGroupGen> PrimitiveGenerators(const std::string& regex = "*") const = 0;
};

class MaterialLogicPoolI
{
    public:
        static constexpr const char* DefaultConstructorName = "GenMaterialPool";
        static constexpr const char* DefaultDestructorName = "DelMaterialPool";

        virtual std::map<std::string, GPUMatGroupGen> MaterialGroupGenerators(const std::string& regex = "*") const = 0;
        virtual std::map<std::string, GPUMatBatchGen> MaterialBatchGenerators(const std::string& regex = "*") const = 0;
};