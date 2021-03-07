#pragma once
#pragma once

#include "LoopingThreadI.h"

class GPUTracerI;

class TracerThread : public LoopingThreadI
{
    private:
        GPUTracerI&     tracer;

        //uint32_t        boolTracerCamera;

    protected:
        bool            InternallyTerminated() const override;
        void            InitialWork() override;
        void            LoopWork() override;
        void            FinalWork() override;

    public:
        // Constructors & Destructor
                        TracerThread(GPUTracerI&);
                        ~TracerThread() = default;

        // All of these functions are delegated to the visor
        // in a thread safe manner
};