#include "TracerThread.h"
#include "TracerI.h"
#include "TracerDistributorI.h"
#include "SceneIO.h"
#include "RayLib/Log.h"

void TracerThread::InitialWork()
{
	tracer.Initialize(seed, logic);
}

void TracerThread::LoopWork()
{
	// Check if data is changed
	CameraPerspective newCam;
	Vector2ui newResolution;
	std::string newScene;
	uint32_t newSample;
	TracerParameters newParams;
	ImageSegment newSegment;
	PixelFormat newFormat;
	double newTime;
	bool timeChanged = time.CheckChanged(newTime);
	bool camChanged = camera.CheckChanged(newCam);
	bool sceneChanged = scene.CheckChanged(newScene);
	bool resolutionChanged = resolution.CheckChanged(newResolution);		
	bool sampleChanged = sample.CheckChanged(newSample);
	bool paramsChanged = parameters.CheckChanged(newParams);
	bool segmentChanged = segment.CheckChanged(newSegment);
	bool imageFormatChanged = pixFormat.CheckChanged(newFormat);

	// Reset Image if images changed
	if(imageFormatChanged)
		tracer.SetImagePixelFormat(newFormat);
	if(segmentChanged)
		tracer.ReportionImage(newSegment.pixelStart,
							  newSegment.pixelCount);
	else if(camChanged || sceneChanged || timeChanged)
		tracer.ResetImage();
		
	if(resolutionChanged)
		tracer.ResizeImage(newResolution);
	if(timeChanged)
		tracer.SetTime(newTime);
	if(sceneChanged)
		tracer.SetScene(newScene);
	if(paramsChanged)
		tracer.SetParams(newParams);
	
	// Initialize Rays
	// Camera ray generation
	tracer.GenerateCameraRays(newCam, newSample);
				
	//=====================//
	//		RenderLoop	   //
	//=====================//
	while(tracer.Continue())
	{			
		tracer.Render();
	}
	tracer.FinishSamples();
	//=====================//
	//	  RenderLoop END   //
	//=====================//

	// Tracer consumed all of the rays 
	// now loop over and generate new camera rays
}

void TracerThread::FinalWork()
{}

bool TracerThread::InternallyTerminated() const
{
	return tracer.IsCrashed();
}

// State Change
void TracerThread::ChangeCamera(const CameraPerspective& persp)
{
	camera = persp;
}

void TracerThread::ChangeScene(const std::string& s)
{
	scene = s;
}

void TracerThread::ChangeResolution(const Vector2ui& res)
{
	resolution = res;
}

void TracerThread::ChangePixelFormat(PixelFormat f)
{
	pixFormat = f;
}

void TracerThread::ChangeSampleCount(uint32_t sampleCount)
{
	sample = sampleCount;
}

void TracerThread::ChangeParams(const TracerParameters& p)
{
	parameters = p;
}

void TracerThread::ChangeTime(double seconds)
{
	time = seconds;
}

void TracerThread::ChangeImageSegment(const Vector2ui& pixelStart,
									  const Vector2ui& pixelCount)
{
	segment = {pixelStart, pixelCount};
}