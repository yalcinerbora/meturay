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

class TestTracerPool final : public TracerPoolI
{
    public:
        // Constructors & Destructor
        TestTracerPool();
        ~TestTracerPool() = default;
};