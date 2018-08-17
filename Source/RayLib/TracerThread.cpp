#include "TracerThread.h"
#include "TracerI.h"
#include "TracerDistributorI.h"
#include "SceneIO.h"
#include "RayLib/Log.h"

void TracerThread::THRDLoop(TracerDistributorI& distributor,
							uint32_t seed)
{
	uint32_t renderCount = 0;

	// Initialize
	tracer.Initialize(seed);

	// Render Loop
	while(!stopSignal)
	{
		// Check if data is changed
		CameraPerspective newCam;
		Vector2ui newResolution;
		std::string newScene;
		uint32_t newSample;
		TracerParameters newParams;
		ImageSegment newSegment;
		double newTime;
		bool timeChanged = time.CheckChanged(newTime);
		bool camChanged = camera.CheckChanged(newCam);
		bool sceneChanged = scene.CheckChanged(newScene);
		bool resolutionChanged = resolution.CheckChanged(newResolution);		
		bool sampleChanged = sample.CheckChanged(newSample);
		bool paramsChanged = parameters.CheckChanged(newParams);
		bool segmentChanged = segment.CheckChanged(newSegment);

		// Reset Image if images changed
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
			

		// Old
		for(int i = 1; i <= 200; i++)
		{
			tracer.ResetImage();
			tracer.GenerateCameraRays(newCam, newSample);
			tracer.HitRays(i);
		}
		METU_LOG("Done");

		// Initialize Rays
		// Camera ray generation
		//tracer.GenerateCameraRays(newCam, newSample);
				
		//=====================//
		//		RenderLoop	   //
		//=====================//
		//while(tracer.RayCount() > 0)
		//{			
		//	tracer.HitRays();

		//	// Check if we are distributed system
		//	if(!distributor.Alone())
		//	{
		//		// Send rays that are not for your responsibility
		//		//tracer.GetMaterialRays();
		//		//distributor.SendMaterialRays();
		//		
		//		// Recieve Rays that are responsible
		//		// by this thread
		//		//distributor.RequestMaterialRays();
		//		//tracer.AddMaterialRays();
		//	}
		//	
		//	// We are ready to bounce rays now
		//	tracer.BounceRays();

		//}
		//=====================//
		//	  RenderLoop END   //
		//=====================//		

		// Image is ready now send according to the callbacks
		if(distributor.ShouldSendImage(renderCount))
		{
			distributor.SendImage(tracer.GetImage(),
								  newResolution,
								  newSegment.pixelStart,
								  newSegment.pixelCount);
		}
		renderCount++;

		// Check at the end of the loop
		// for the signals
		{
			std::unique_lock<std::mutex> lock(mutex);
			conditionVar.wait(lock, [&]
			{
				return stopSignal || !pauseSignal;
			});
		}
	}
	// Do cleanup
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