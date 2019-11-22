#pragma once

#include "TracerLib/TracerLogicGenerator.h"
#include "TracerLib/TracerLogicPools.h"


class TestMaterialPool final : public MaterialLogicPoolI
{
    private:
        std::map<std::string, GPUMatGroupGen>   matGroupGenerators;
        std::map<std::string, GPUMatBatchGen>   matBatchGenerators;

    public:
        // Constructors & Destructor
                                                TestMaterialPool();
                                                ~TestMaterialPool() = default;

        // Interface
        std::map<std::string, GPUMatGroupGen>   MaterialGroupGenerators(const std::string& regex = "0") const override;
        std::map<std::string, GPUMatBatchGen>   MaterialBatchGenerators(const std::string& regex = "0") const override;
};

class BasicTracerLogicGenerator final : public TracerLogicGenerator
{
    private:

    protected:
    public:
        // Constructors & Destructor
                            BasicTracerLogicGenerator();
                            ~BasicTracerLogicGenerator() = default;
};