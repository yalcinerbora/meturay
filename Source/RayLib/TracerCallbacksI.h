#pragma once

#include <vector>

#include "Vector.h"
#include "Types.h"
#include "HitStructs.h"
#include "Constants.h"

struct TracerError;
struct TracerOptions;
struct AnalyticData;

struct CameraPerspective;

class TracerCallbacksI
{
    public:
        virtual         ~TracerCallbacksI() = default;

        virtual void    SendLog(const std::string) = 0;
        virtual void    SendError(TracerError) = 0;
        virtual void    SendAnalyticData(AnalyticData) = 0;
        virtual void    SendImage(const std::vector<Byte> data,
                                  PixelFormat, int sampleCount,
                                  Vector2i start = Zero2i,
                                  Vector2i end = BaseConstants::IMAGE_MAX_SIZE) = 0;
        virtual void    SendAccelerator(HitKey key, const std::vector<Byte> data) = 0;
        virtual void    SendBaseAccelerator(const std::vector<Byte> data) = 0;
};

class TracerCommandsI
{
    public:
        virtual         ~TracerCommandsI() = default;
};