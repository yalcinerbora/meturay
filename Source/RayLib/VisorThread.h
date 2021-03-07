#pragma once

#include "LoopingThreadI.h"

class VisorI;

class VisorThread : public LoopingThreadI
{
    private:
        VisorI&         visor;

    protected:
        bool            InternallyTerminated() const override;
        void            InitialWork() override;
        void            LoopWork() override;
        void            FinalWork() override;

    public:
        // Constructors & Destructor
                        VisorThread(VisorI&);
                        ~VisorThread() = default;

        // All of these functions are delegated to the visor
        // in a thread safe manner




        // Main Thread Call
        void            ProcessInputs();


};