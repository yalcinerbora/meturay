#pragma once

#include <string>

struct CPUCamera;
struct TracerCommonOptions;

enum class ImageType;

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

        virtual void        ChangeCamera(const CPUCamera) = 0;
        virtual void        ChangeCamera(const unsigned int) = 0;

        // Control Flow of the Simulation
        virtual void        StartStopTrace(const bool) = 0;
        virtual void        PauseContTrace(const bool) = 0;

        virtual void        SetTimeIncrement(const double) = 0;
        
        virtual void        SaveImage() = 0;                        // Default Image Save
        virtual void        SaveImage(const std::string& path,      // Location folder
                                      const std::string& fileName,  // without extension
                                      ImageType,                    // Determines extension etc.
                                      bool overwriteFile) = 0;      // Do overwrite 
                                                                    // (if false new file will be created with suffix integer)
};