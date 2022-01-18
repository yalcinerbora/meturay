#include "TracerThread.h"
#include "TracerSystemI.h"
#include "GPUTracerI.h"
#include "TracerCallbacksI.h"
#include "Log.h"
#include "VisorTransform.h"
#include "SceneError.h"

#include "RayLib/CPUTimer.h"
#include "RayLib/ImageIOError.h"

TracerThread::TracerThread(TracerSystemI& t,
                           const TracerOptions& opts,
                           const TracerParameters& params,
                           TracerCallbacksI& tracerCallbacks,
                           const std::string& tracerTypeName)
    : tracerSystem(t)
    , tracer(nullptr, nullptr)
    , currentScene(nullptr)
    , tracerOptions(opts)
    , tracerParameters(params)
    , tracerTypeName(tracerTypeName)
    , tracerCallbacks(tracerCallbacks)
    , tracerCrashSignal(false)
    , isPrevStopped(false)
{}

bool TracerThread::InternallyTerminated() const
{
    return tracerCrashSignal;
}

void TracerThread::InitialWork()
{
    // No initial work for tracer

    // TODO: CHANGE THIS LATERR
    isSceneCameraActive = true;
    sceneCam = 0;
}

void TracerThread::LoopWork()
{
    bool imageAlreadyChanged = false;
    bool newSceneGenerated = false;
    bool reallocateTracer = false;

    // First check that the scene is changed
    std::u8string newScene;
    if(currentScenePath.CheckChanged(newScene) || isPrevStopped)
    {
        newSceneGenerated = true;

        SceneLoadFlags flags;
        if(tracerParameters.forceOptiX)
            flags |= SceneLoadFlagType::FORCE_OPTIX_ACCELS;

        // First Generate Scene
        tracerSystem.GenerateScene(currentScene, newScene, flags);
        // We need to re-create tracer
        // since it is scene dependent
        // First deallocate tracer
        tracer = GPUTracerPtr(nullptr, nullptr);
        reallocateTracer = true;
    }
    // Check scene time and regenerate scene if new scene is requested
    // Check if the time is changed
    SceneError sError = SceneError::OK;
    double newTime;
    bool timeChanged = currentTime.CheckChanged(newTime);
    // Generate Scene (time change is implicit)
    if(newSceneGenerated)
    {
        if((sError = currentScene->LoadScene(newTime)) != SceneError::OK)
        {
            PrintErrorAndSignalTerminate(sError);
            return;
        }
    }
    // Change time if required
    else if(timeChanged &&
            ((sError = currentScene->ChangeTime(newTime)) != SceneError::OK))
    {
        PrintErrorAndSignalTerminate(sError);
        return;
    }

    // Send new camera count to the visor(s)
    if(newSceneGenerated)
    {
        tracerCallbacks.SendCurrentSceneCameraCount(currentScene->CameraCount());
    }

    // Now scene is reorganized
    // Recreate tracer if necessary
    if(reallocateTracer)
    {
        // Now create and check for error
        TracerError tError = TracerError::OK;
        if((tError = RecreateTracer()) != TracerError::OK)
        {
            PrintErrorAndSignalTerminate(tError);
            return;
        }
        // Reset the Image as well
        Vector2i newRes, newStart, newEnd;
        resolution.CheckChanged(newRes);
        imgPortionStart.CheckChanged(newStart);
        imgPortionEnd.CheckChanged(newEnd);

        tracer->SetImagePixelFormat(PixelFormat::RGBA_FLOAT);
        tracer->ResizeImage(newRes);
        tracer->ReportionImage(newStart, newEnd);

        imageAlreadyChanged = true;
    }

    // TODO: wtf is this?
    if(!tracer) return;

    // Check if image is changed
    Vector2i newRes;
    if(!imageAlreadyChanged && resolution.CheckChanged(newRes))
    {
        tracer->ResizeImage(newRes);
    }
    Vector2i newStart;
    Vector2i newEnd;
    bool startChanged = imgPortionStart.CheckChanged(newStart);
    bool endChanged = imgPortionEnd.CheckChanged(newEnd);
    if(startChanged || endChanged)
    {
        tracer->ReportionImage(newStart, newEnd);
    }
    // Generate work according to the camera that is being selected
    if(isSceneCameraActive.Get())
    {
        uint32_t cam;
        if(sceneCam.CheckChanged(cam))
            tracer->ResetImage();
        tracer->GenerateWork(cam);
    }
    else
    {
        VisorTransform vt;
        if(visorTransform.CheckChanged(vt))
            tracer->ResetImage();
        tracer->GenerateWork(vt, sceneCam.Get());
    }

    try
    {
        // Exhaust all the generated work
        while(tracer->Render());

        // Finalize the Works
        // (send the generated image to the visor etc.)
        tracer->Finalize();
    }
    catch(const TracerException& e)
    {
        PrintErrorAndSignalTerminate<TracerError>(e);
        tracerCallbacks.SendError(e);
        tracerCallbacks.SendCrashSignal();
        return;
    }
    catch(const ImageIOException& e)
    {
        PrintErrorAndSignalTerminate<ImageIOError>(e);
        // Convert it to a Tracer Error
        TracerError err = TracerError(TracerError::TRACER_INTERNAL_ERROR,
                                      static_cast<ImageIOError>(e));

        tracerCallbacks.SendError(err);
        tracerCallbacks.SendCrashSignal();
        return;
    }

    // Set previously stopped to false since we cycled once
    isPrevStopped = false;
}

void TracerThread::FinalWork()
{
    // In final work (after loop),
    // just set previously stopped
    isPrevStopped = true;
    // Clear the scene
    tracer = nullptr;
    tracerSystem.ClearScene();
    currentScene = nullptr;

    // Everything else should destroy gracefully
}

TracerError TracerThread::RecreateTracer()
{
    TracerError tError = TracerError::OK;
    if((tError = tracerSystem.GenerateTracer(tracer,
                                             tracerParameters,
                                             tracerOptions,
                                             tracerTypeName)) != TracerError::OK)
        return tError;

    tracer->AttachTracerCallbacks(tracerCallbacks);

    if((tError = tracer->Initialize()) != TracerError::OK)
    {
        // Remove the tracer
        tracer = nullptr;
        return tError;
    }
    return TracerError::OK;
}

void TracerThread::SetScene(std::u8string sceneName)
{
    currentScenePath = sceneName;
}

void TracerThread::ChangeTime(double t)
{
    currentTime = t;
}

void TracerThread::IncreaseTime(double t)
{
    double nextTime = currentTime.Get() + t;
    if(currentScene)
    {
        nextTime = std::min(currentScene->MaxSceneTime(), nextTime);
        currentTime = nextTime;
    }
}

void TracerThread::DecreaseTime(double t)
{
    double nextTime = currentTime.Get() - t;

    if(currentScene)
    {
        nextTime = std::max(0.0, nextTime);
        currentTime = nextTime;
    }
    currentTime = currentTime.Get() - t;
}

void TracerThread::ChangeCamera(unsigned int sceneCamId)
{
    // If multiple visors set a new camera
    // (i.e. one visor sets VisorCam other sets scene cam)
    // simultaneously this will fail
    // However system is designed for one Visor many Tracer in mind
    isSceneCameraActive = true;
    sceneCam = sceneCamId;
}

void TracerThread::ChangeTransform(VisorTransform t)
{
    // Same as above
    isSceneCameraActive = false;
    visorTransform = t;
}

void TracerThread::StartStopTrace(bool start)
{
    if(start)
        Start();
    else
        Stop();
}

void TracerThread::PauseContTrace(bool pause)
{
    // Just call pause here
    // This will only pause the system when tracer finishes its job
    // This should be ok since tracer should be working incrementally
    // anyway in order to prevent OS GPU Hang driver restart
    Pause(pause);
}

void TracerThread::SetImageResolution(Vector2i r)
{
    resolution = r;
}

void TracerThread::SetImagePortion(Vector2i start, Vector2i end)
{
    imgPortionStart = start;
    imgPortionEnd = end;
}