#pragma once

#include <string>
#include "VisorTransform.h"

struct TracerCommonOptions;

enum class ImageType;

class CommandCallbacksI
{
    public:
        virtual             ~CommandCallbacksI() = default;

        // Fundamental Scene Commands
        // Current Scene and Current Time on that Scene
        virtual void        ChangeScene(const std::u8string) = 0;
        virtual void        ChangeTime(const double) = 0;
        virtual void        IncreaseTime(const double) = 0;
        virtual void        DecreaseTime(const double) = 0;

        virtual void        ChangeCamera(const VisorTransform) = 0;
        virtual void        ChangeCamera(const unsigned int) = 0;

        // Control Flow of the Simulation
        virtual void        StartStopTrace(const bool) = 0;
        virtual void        PauseContTrace(const bool) = 0;
};

class EmptyCommandCallback : public CommandCallbacksI
{
    void    ChangeScene(const std::u8string) override {}
    void    ChangeTime(const double) override {}
    void    IncreaseTime(const double) override {}
    void    DecreaseTime(const double) override {}

    void    ChangeCamera(const VisorTransform) override {}
    void    ChangeCamera(const unsigned int) override {}
    void    StartStopTrace(const bool) override {}
    void    PauseContTrace(const bool) override {}
};