#pragma once
/**

Non-distributed version of system

It directly delegates a MVisor commands to a single MTracer
and vice versa.

*/

#include "NodeI.h"
#include "VisorCallbacksI.h"
#include "TracerCallBacksI.h"

class VisorI;
class TracerI;

class SelfNode
    : public VisorCallbacksI
    , public TracerCallbacksI
    , public NodeI
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
        void        ChangeScene(const std::string) override;
        void        ChangeTime(const double) override;
        void        IncreaseTime(const double) override;
        void        DecreaseTime(const double) override;
        void        ChangeCamera(const CameraPerspective) override;
        void        ChangeCamera(const unsigned int) override;
        void        ChangeOptions(const TracerOptions) override;
        void        StartStopTrace(const bool) override;
        void        PauseContTrace(const bool) override;
        void        SetTimeIncrement(const double) override;
        void        SaveImage() override;
        void        SaveImage(const std::string& path,
                              const std::string& fileName,
                              ImageType,
                              bool overwriteFile) override;

        void        WindowMinimizeAction(bool minimized) override;
        void        WindowCloseAction() override;

        // From Tracer Callbacks
        void        SendLog(const std::string) override;
        void        SendError(TracerError) override;
        void        SendAnalyticData(AnalyticData) override;
        void        SendImage(const std::vector<Byte> data,
                              PixelFormat, size_t offset,
                              Vector2i start = Zero2i,
                              Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void        SendAccelerator(HitKey key, const std::vector<Byte> data) override;
        void        SendBaseAccelerator(const std::vector<Byte> data) override;

        // From Node Interface
        NodeError   Initialize() override;
        void        Work() override;
};