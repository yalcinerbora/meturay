#include "TracerThread.h"
#include "TracerSystemI.h"
#include "GPUTracerI.h"
#include "Log.h"
#include "VisorCamera.h"
#include "SceneError.h"

TracerThread::TracerThread(TracerSystemI& t,
                           const TracerOptions& opts,
                           const TracerParameters& params,
                           TracerCallbacksI& tracerCallbacks,
                           const std::string& tracerTypeName)
    : tracerSystem(t)
    , tracer(nullptr, nullptr)
    , tracerOptions(opts)
    , tracerParameters(params)
    , tracerTypeName(tracerTypeName)
    , tracerCallbacks(tracerCallbacks)
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
    if(currentScenePath.CheckChanged(newScene))
    {
        newSceneGenerated = true;
        // First Generate Scene
        tracerSystem.GenerateScene(currentScene, newScene);
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
    if(currentTime.CheckChanged(newTime))
    {
        if(newSceneGenerated &&
           (sError = currentScene->LoadScene(newTime)) != SceneError::OK)
        {
            PrintErrorAndSignalTerminate(sError);
            return;
        }            
        else if((sError = currentScene->ChangeTime(newTime)) != SceneError::OK)
        {
            PrintErrorAndSignalTerminate(sError);
            return;
        }
    }
    else if(newSceneGenerated &&
            (sError = currentScene->LoadScene(newTime)) != SceneError::OK)
    {
        PrintErrorAndSignalTerminate(sError);
        return;
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
        // Reset the Image aswell        
        tracer->SetImagePixelFormat(PixelFormat::RGBA_FLOAT);
        tracer->ResizeImage(resolution.Get());
        tracer->ReportionImage(imgPortionStart.Get(),
                               imgPortionEnd.Get());

        imageAlreadyChanged = true;
    }

    // Check if image is changed
    Vector2i newRes;
    if(!imageAlreadyChanged && resolution.CheckChanged(newRes))
    {
        tracer->ResizeImage(resolution.Get());
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
        {
            // Camera changed reset image
            tracer->ResetImage();
        }

        tracer->GenerateWork(sceneCam.Get());
    }
    else tracer->GenerateWork(visorCam.Get());

    // Exaust all the generated work
    while(tracer->Render());

    // Finalaze the Works
    // (send the generated image to the visor etc.)
    tracer->Finalize();
}

void TracerThread::FinalWork()
{
    // No final work for tracer
    // Eveything should destroy gracefully
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
        return tError;
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

void TracerThread::ChangeCamera(VisorCamera cam)
{
    // Same as above
    isSceneCameraActive = false;
    visorCam = cam;
}

void TracerThread::StartStopTrace(bool start)
{
    //// This command only be callsed when the tracer thread is
    //// already available
    //// Start(), Stop() functions cannot be used here we need to utilize new
    //// condition_var mutex pair for this


    //if(start)
    //    Start();
    //else
    //    Stop();
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