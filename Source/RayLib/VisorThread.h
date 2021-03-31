#pragma once

#include <vector>
#include "LoopingThreadI.h"
#include "Vector.h"
#include "Types.h"
#include "Constants.h"

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
        void            AccumulateImagePortion(const std::vector<Byte> data,
                                               PixelFormat, size_t offset,
                                               Vector2i start = Zero2i,
                                               Vector2i end = BaseConstants::IMAGE_MAX_SIZE);

        // Main Thread Call
        void            ProcessInputs();
};