#pragma once

#include <string>

#include "Vector.h"
#include "Types.h"
#include "Camera.h"

#include "TracerStructs.h"

struct TracerStatus
{
    std::string                 currentScene;           // Current scene that is being render
    unsigned int                cameraCount;            // Total camera count on that scene

    unsigned int                latestSceneCamId;       // Latest camera that has been used
                                                        // from the scene (for switching from that)
    CameraPerspective           currentCamera;          // Current Camera (as data)

    Vector2i                    currentRes;             // Current Resolution of the scene;
    PixelFormat                 currentPixelFormat;     // Pixel format of the image that is being generated

    double                      currentTime;            // Current animation time point that is being rendered

    bool                        pausedCont;             // Pause-Cont.
    bool                        startedStop;            // Start-Stop

    TracerOptions               currentTOpts;           // Current Tracer Options
    TracerParameters            tracerParameters;
};