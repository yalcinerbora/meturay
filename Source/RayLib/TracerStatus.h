#pragma once

#include <string>

#include "Vector.h"
#include "Types.h"

#include "TracerStructs.h"

struct TracerStatus
{
    std::u8string               currentScene;           // Current scene that is being rendered
    unsigned int                cameraCount;            // Total camera count on that scene

    unsigned int                latestSceneCamId;       // Latest camera that has been used
                                                        // from the scene (for switching from that)

    Vector2i                    currentRes;             // Current Resolution of the scene;
    PixelFormat                 currentPixelFormat;     // Pixel format of the image that is being generated

    double                      currentTime;            // Current animation time point that is being rendered

    bool                        pausedCont;             // Pause-Cont.
    bool                        startedStop;            // Start-Stop

    TracerParameters            tracerParameters;
};