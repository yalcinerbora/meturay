#pragma once
/**

Non-distributed version of system

It directly delegates a MVisor commands to a single MTracer
and vice versa.

*/

#include "NodeI.h"
#include "VisorCallbacksI.h"
#include "TracerCallbacksI.h"
#include "TracerThread.h"

class VisorI;
class TracerSystemI;

class SelfNode
    : public VisorCallbacksI
    , public TracerCallbacksI
    , public NodeI
{
    private:
        // Threads
        TracerThread    tracerThread;
        VisorI&         visor;

        std::atomic_bool tracerCrashed;


    protected:
    public:
        // Constructor & Destructor
                    SelfNode(VisorI&, TracerSystemI&,
                             const TracerOptions&,
                             const TracerParameters&,
                             const std::string& tracerTypeName,
                             const Vector2i& resolution);
                    ~SelfNode() = default;

        // From Command Callbacks
        void        ChangeScene(std::u8string) override;
        void        ChangeTime(double) override;
        void        IncreaseTime(double) override;
        void        DecreaseTime(double) override;
        void        ChangeCamera(VisorTransform) override;
        void        ChangeCamera(unsigned int) override;
        void        StartStopTrace(bool) override;
        void        PauseContTrace(bool) override;

        void        WindowMinimizeAction(bool minimized) override;
        void        WindowCloseAction() override;

        // From Tracer Callbacks
        void        SendCrashSignal() override;
        void        SendLog(const std::string) override;
        void        SendError(TracerError) override;
        void        SendAnalyticData(TracerAnalyticData) override;
        void        SendSceneAnalyticData(SceneAnalyticData) override;
        void        SendImageSectionReset(Vector2i start = Zero2i,
                                          Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
        void        SendImage(const std::vector<Byte> data,
                              PixelFormat, size_t offset,
                              Vector2i start = Zero2i,
                              Vector2i end = BaseConstants::IMAGE_MAX_SIZE) override;
         void       SendCurrentOptions(TracerOptions) override;
         void       SendCurrentParameters(TracerParameters) override;
         void       SendCurrentTransform(VisorTransform) override;
         void       SendCurrentSceneCameraCount(uint32_t) override;

        // From Node Interface
        NodeError   Initialize() override;
        void        Work() override;
};