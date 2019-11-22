#pragma once

#include <map>
#include <string>

#include "DefaultTypeGenerators.h"

class AcceleratorLogicPoolI
{
    public:
        static constexpr const char* PoolConstructorName = "GenAcceleratorPool";
        static constexpr const char* PoolDestructorName = "DelAcceleratorPool";

        virtual std::map<std::string, GPUAccelGroupGen> AcceleratorGroupGenerators(const std::string& regex = "0") const = 0;
        virtual std::map<std::string, GPUAccelBatchGen> AcceleratorBatchGenerators(const std::string& regex = "0") const = 0;
};

class BaseAcceleratorLogicPoolI
{
    public:
        static constexpr const char* PoolConstructorName = "GenBaseAcceleratorPool";
        static constexpr const char* PoolDestructorName = "DelBaseAcceleratorPool";

        virtual std::map<std::string, GPUBaseAccelGen> BaseAcceleratorGenerators(const std::string& regex = "0") const = 0;
};

class PrimitiveLogicPoolI
{
    public:
        static constexpr const char* PoolConstructorName = "GenPrimitivePool";
        static constexpr const char* PoolDestructorName = "DelPrimitivePool";

        virtual std::map<std::string, GPUPrimGroupGen> PrimitiveGenerators(const std::string& regex = "0") const = 0;
};

class MaterialLogicPoolI
{
    public:
        static constexpr const char* PoolConstructorName = "GenMaterialPool";
        static constexpr const char* PoolDestructorName = "DelMaterialPool";

        virtual std::map<std::string, GPUMatGroupGen> MaterialGroupGenerators(const std::string& regex = "0") const = 0;
        virtual std::map<std::string, GPUMatBatchGen> MaterialBatchGenerators(const std::string& regex = "0") const = 0;
};