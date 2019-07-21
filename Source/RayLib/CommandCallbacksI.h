#pragma once

#include <string>

struct CameraPerspective;
struct TracerOptions;

class CommandCallbacksI
{
    public:
        virtual             ~CommandCallbacksI() = default;

        // Fundamental Scene Commands
        // Current Scene and Current Time on that Scene
        virtual void        ChangeScene(const std::string) = 0;
        virtual void        ChangeTime(const double) = 0;
        virtual void        IncreaseTime(const double) = 0;
        virtual void        DecreaseTime(const double) = 0;

        virtual void        ChangeCamera(const CameraPerspective) = 0;
        virtual void        ChangeCamera(const unsigned int) = 0;
        virtual void        ChangeOptions(const TracerOptions) = 0;

        // Control Flow of the Simulation
        virtual void        StartStopTrace(const bool) = 0;
        virtual void        PauseContTrace(const bool) = 0;
};