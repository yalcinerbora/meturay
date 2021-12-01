#pragma once

#include <vector>

#include "Vector.h"
#include "Types.h"
#include "HitStructs.h"
#include "Constants.h"

struct TracerError;
struct TracerAnalyticData;
struct SceneAnalyticData;
struct TracerParameters;
struct VisorTransform;

class TracerOptions;

class TracerCallbacksI
{
    public:
        virtual         ~TracerCallbacksI() = default;

        virtual void    SendCrashSignal() = 0;
        virtual void    SendLog(const std::string) = 0;
        virtual void    SendError(TracerError) = 0;
        virtual void    SendAnalyticData(TracerAnalyticData) = 0;
        virtual void    SendSceneAnalyticData(SceneAnalyticData) = 0;
        virtual void    SendImageSectionReset(Vector2i start = Zero2i,
                                              Vector2i end = BaseConstants::IMAGE_MAX_SIZE) = 0;
        virtual void    SendImage(const std::vector<Byte> data,
                                  PixelFormat, size_t offset,
                                  Vector2i start = Zero2i,
                                  Vector2i end = BaseConstants::IMAGE_MAX_SIZE) = 0;
        virtual void    SendCurrentOptions(TracerOptions) = 0;
        virtual void    SendCurrentParameters(TracerParameters) = 0;
        virtual void    SendCurrentTransform(VisorTransform) = 0;
        virtual void    SendCurrentSceneCameraCount(uint32_t) = 0;
};