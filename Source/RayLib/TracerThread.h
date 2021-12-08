#pragma once
#pragma once

#include "LoopingThreadI.h"
#include "TracerStructs.h"
#include "VisorTransform.h"
#include "ThreadVariable.h"
#include "Constants.h"
#include "TracerOptions.h"
#include "Log.h"

class TracerSystemI;
class GPUSceneI;
class TracerCallbacksI;

class TracerThread : public LoopingThreadI
{
    private:
        // Tracer system of this thread
        // This and the tracer are not thread safe
        // these class accomplishes that
        TracerSystemI&              tracerSystem;
        GPUTracerPtr                tracer;

        // Current Scene
        GPUSceneI*                  currentScene;

        // Tracer Parameters & Options
        // In case of reloading of tracer
        TracerOptions               tracerOptions;
        TracerParameters            tracerParameters;
        std::string                 tracerTypeName;
        TracerCallbacksI&           tracerCallbacks;

        // Internal Bool for checking if tracer is crashed
        // This technically is not needed to be an atomic
        // but just to be sure, it is.
        //
        // If Tracer uses multiple threads (which it should not
        // since it is GPU based) this will come in handy
        std::atomic_bool            tracerCrashSignal;
        // Not initialized Bool
        // Only accessed from the thread
        // If system is stopped and started again
        // All objects needs to be re-initialized (re-constructed)
        bool                        isPrevStopped;

        // State variables
        // Camera Related
        ThreadVariable<bool>            isSceneCameraActive;
        ThreadVariable<VisorTransform>  visorTransform;
        ThreadVariable<uint32_t>        sceneCam;

        // Scene Related
        ThreadVariable<std::u8string>   currentScenePath;
        // Scene Time Related
        ThreadVariable<double>          currentTime;

        // Image Related
        ThreadVariable<Vector2i>        resolution;
        ThreadVariable<Vector2i>        imgPortionStart;
        ThreadVariable<Vector2i>        imgPortionEnd;

    protected:
        bool                InternallyTerminated() const override;
        void                InitialWork() override;
        void                LoopWork() override;
        void                FinalWork() override;

        TracerError         RecreateTracer();

        template<class Error>
        void                PrintErrorAndSignalTerminate(Error);

    public:
        // Constructors & Destructor
                        TracerThread(TracerSystemI&,
                                     const TracerOptions&,
                                     const TracerParameters&,
                                     TracerCallbacksI&,
                                     const std::string& tracerTypeName);
                        ~TracerThread() = default;

        void            SetScene(std::u8string sceneName);
        void            ChangeTime(double);
        void            IncreaseTime(double);
        void            DecreaseTime(double);
        void            ChangeCamera(unsigned int);
        void            ChangeTransform(VisorTransform);
        void            StartStopTrace(bool);
        void            PauseContTrace(bool);

        void            SetImageResolution(Vector2i);
        void            SetImagePortion(Vector2i start = Zero2i,
                                        Vector2i end = BaseConstants::IMAGE_MAX_SIZE);
};

template<class Error>
void TracerThread::PrintErrorAndSignalTerminate(Error err)
{
    METU_ERROR_LOG("Tracer Thread, {:s}",
                   static_cast<std::string>(err));
    tracerCrashSignal = true;
}