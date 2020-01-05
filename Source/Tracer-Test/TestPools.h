#pragma once

#include "TracerLib/TracerLogicGenerator.h"
#include "TracerLib/TracerLogicPools.h"


class TestMaterialPool final : public MaterialLogicPoolI
{
    public:
        // Constructors & Destructor
        TestMaterialPool();
        ~TestMaterialPool() = default;
};

class TestTracerLogicPool final : public TracerLogicPoolI
{
    public:
        // Constructors & Destructor
        TestTracerLogicPool();
        ~TestTracerLogicPool() = default;
};