#pragma once
/**

Non-distributed version of system

It directly delegates a MVisor commands to a single MTracer
and vice versa.

*/

#include "VisorCallbacksI.h"
#include "TracerCallBacksI.h"

class VisorI;
class TracerI;

class SelfNode
    : public VisorCallbacksI
    , public TracerCallbacksI
{
    private:
        VisorI&     visor;
        TracerI&    tracer;

    protected:
    public:
        // Constructor & Destructor
                    SelfNode(VisorI&, TracerI&);
                    ~SelfNode() = default;

        // From Command Callbacks
        void        SendScene(const std::string) override;
        void        SendTime(const double) override;
        void        IncreaseTime(const double) override;
        void        DecreaseTime(const double) override;
        void        SendCamera(const CameraPerspective) override;
        void        SendOptions(const TracerOptions) override;
        void        StartStopTrace(const bool) override;
        void        PauseContTrace(const bool) override;

        void        WindowMinimizeAction(bool minimized) override;
        void        WindowCloseAction() override;

        // From Tracer Callbacks
        void        SendLog(const std::string) override;
        void        SendError(TracerError) override;
        void        SendAnalyticData(AnalyticData) override;
        void        SendImage(const std::vector<Byte> data,
                              PixelFormat, int sampleCount,
                              Vector2i start = Zero2i,
                              Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void        SendAccelerator(HitKey key, const std::vector<Byte> data) override;
        void        SendBaseAccelerator(const std::vector<Byte> data) override;
};