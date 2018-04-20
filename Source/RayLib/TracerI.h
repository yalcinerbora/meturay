#pragma once

/**

Tracer Interface

Main Interface for Tracer DLLs. Only GPU tracer will be
implemented, still tracer is interfaced for further implementations

Tracer Interface is a threaded interface (which means that it repsesents a thread)
which does send commands to GPU to do ray tracing 
(it is responsible for utilizing all GPUs on the computer). 



*/

#include <cstdint>
#include <future>
#include "Vector.h"
#include "RayHitStructs.h"
#include "WorkerThread.h"

class SceneI;
struct CameraPerspective;

struct TracerParameters
{
	uint32_t depth;
};

class TracerI
{
	public:
		virtual							~TracerI() = default;

		// Main Thread Only Calls
		virtual void					Initialize() = 0;

		// Main Calls
		virtual std::future<void>		AssignScene(const SceneI&) = 0;
		virtual std::future<void>		SetParams(const TracerParameters&) = 0;
	
		// Initial Generations
		virtual std::future<void>		GenerateSceneAccelerator() = 0;
		virtual std::future<void>		GenerateAccelerator(uint32_t objId) = 0;
		virtual std::future<void>		AssignImageSegment(const Vector2ui& pixelStart,
														   const Vector2ui& pixelEnd) = 0;
		// Material Related
		// Main memory bottleneck is materials.
		// Tracers are designed around this bottlenech considering GPU memory limitations.
		// A tracer will be assigned with a specific material and those rays that hits
		// to that mat will be transferred to that tracer
		virtual std::future<void>		AssignAllMaterials() = 0;
		virtual std::future<void>		AssignMaterial(uint32_t matId) = 0;
		virtual std::future<void>		LoadMaterial(uint32_t matId) = 0;
		virtual std::future<void>		UnloadMaterial(uint32_t matId) = 0;

		// Rendering
		// Loop HitRays/BounceRays until ray count is zero
		// Transfer Material rays between tracer nodes using Get/AddMaterialRays
		virtual std::future<void>		GenerateCameraRays(const CameraPerspective& camera,
														   const uint32_t samplePerPixel) = 0;		
		virtual std::future<void>		HitRays() = 0;		
		virtual std::future<void>		GetMaterialRays(const RayRecodCPU&, uint32_t rayCount, uint32_t matId) = 0;
		virtual std::future<void>		AddMaterialRays(const ConstRayRecodCPU&, uint32_t rayCount, uint32_t matId) = 0;
		virtual std::future<void>		BounceRays() = 0;
		virtual std::future<uint32_t>	RayCount() = 0;
};

// Threading wrapper Interface for Tracer
class TracerTI : public TracerI
{
	private:
	protected:
		WorkerThread			thread;

		virtual void			THRDAssignScene(const SceneI&) = 0;
		virtual void			THRDSetParams(const TracerParameters&) = 0;

		virtual void			THRDGenerateSceneAccelerator() = 0;
		virtual void			THRDGenerateAccelerator(uint32_t objId) = 0;
		virtual void			THRDAssignImageSegment(const Vector2ui& pixelStart,
															   const Vector2ui& pixelEnd) = 0;

		virtual void			THRDAssignAllMaterials() = 0;
		virtual void			THRDAssignMaterial(uint32_t matId) = 0;
		virtual void			THRDLoadMaterial(uint32_t matId) = 0;
		virtual void			THRDUnloadMaterial(uint32_t matId) = 0;

		virtual void			THRDGenerateCameraRays(const CameraPerspective& camera,
												   const uint32_t samplePerPixel) = 0;
		virtual void			THRDHitRays() = 0;		
		virtual void			THRDGetMaterialRays(const RayRecodCPU&, uint32_t rayCount, uint32_t matId) = 0;
		virtual void			THRDAddMaterialRays(const ConstRayRecodCPU&, uint32_t rayCount, uint32_t matId) = 0;
		virtual void			THRDBounceRays() = 0;
		virtual uint32_t		THRDRayCount() = 0;

	public:
		// Constructors & Destructor
								TracerTI();
								TracerTI(const TracerTI&) = delete;
		TracerTI&				operator=(const TracerTI&) = delete;
								~TracerTI();

		std::future<void>		AssignScene(const SceneI&) override;
		std::future<void>		SetParams(const TracerParameters&) override;
	
		std::future<void>		GenerateSceneAccelerator() override;
		std::future<void>		GenerateAccelerator(uint32_t objId) override;
		std::future<void>		AssignImageSegment(const Vector2ui& pixelStart,
												   const Vector2ui& pixelEnd) override;

		std::future<void>		AssignAllMaterials() override;
		std::future<void>		AssignMaterial(uint32_t matId) override;
		std::future<void>		LoadMaterial(uint32_t matId) override;
		std::future<void>		UnloadMaterial(uint32_t matId) override;

		std::future<void>		GenerateCameraRays(const CameraPerspective& camera,
												   const uint32_t samplePerPixel) override;
		std::future<void>		HitRays() override;
		std::future<void>		GetMaterialRays(const RayRecodCPU&, uint32_t rayCount, uint32_t matId) override;
		std::future<void>		AddMaterialRays(const ConstRayRecodCPU&, uint32_t rayCount, uint32_t matId) override;
		std::future<void>		BounceRays() override;
		std::future<uint32_t>	RayCount() override;
};

TracerTI::TracerTI()
{
	thread.Start();
}

TracerTI::~TracerTI()
{
	thread.Stop();
}

inline std::future<void> TracerTI::AssignScene(const SceneI& s)
{
	return thread.AddWork(&TracerTI::THRDAssignScene, this, s);
}

inline std::future<void> TracerTI::SetParams(const TracerParameters& p)
{
	return thread.AddWork(&TracerTI::THRDSetParams, this, p);
}

inline std::future<void> TracerTI::GenerateSceneAccelerator()
{
	return thread.AddWork(&TracerTI::THRDGenerateSceneAccelerator, this);
}

inline std::future<void> TracerTI::GenerateAccelerator(uint32_t objId)
{
	return thread.AddWork(&TracerTI::THRDGenerateAccelerator, this, objId);
}

inline std::future<void> TracerTI::AssignImageSegment(const Vector2ui& pixelStart,
											   const Vector2ui& pixelEnd)
{
	return thread.AddWork(&TracerTI::THRDAssignImageSegment, this, 
						  pixelStart,
						  pixelEnd);
}

inline std::future<void> TracerTI::AssignAllMaterials()
{
	return thread.AddWork(&TracerTI::THRDAssignAllMaterials, this);
}

inline std::future<void> TracerTI::AssignMaterial(uint32_t matId)
{
	return thread.AddWork(&TracerTI::THRDAssignMaterial, this, matId);
}

inline std::future<void> TracerTI::LoadMaterial(uint32_t matId)
{
	return thread.AddWork(&TracerTI::THRDLoadMaterial, this, matId);
}

inline std::future<void> TracerTI::UnloadMaterial(uint32_t matId)
{
	return thread.AddWork(&TracerTI::THRDUnloadMaterial, this, matId);
}

inline std::future<void> TracerTI::GenerateCameraRays(const CameraPerspective& camera,
											   const uint32_t samplePerPixel)
{
	return thread.AddWork(&TracerTI::THRDGenerateCameraRays, this, 
						  camera, samplePerPixel);
}

inline std::future<void> TracerTI::HitRays()
{
	return thread.AddWork(&TracerTI::THRDHitRays, this);
}

inline std::future<void> TracerTI::GetMaterialRays(const RayRecodCPU& rec, uint32_t rayCount, uint32_t matId)
{
	return thread.AddWork(&TracerTI::THRDGetMaterialRays, this, 
						  rec, rayCount, matId);
}

inline std::future<void> TracerTI::AddMaterialRays(const ConstRayRecodCPU& rec, uint32_t rayCount, uint32_t matId)
{
	return thread.AddWork(&TracerTI::THRDAddMaterialRays, this,
						  rec, rayCount, matId);
}

inline std::future<void> TracerTI::BounceRays()
{
	return thread.AddWork(&TracerTI::THRDBounceRays, this);
}

inline std::future<uint32_t> TracerTI::RayCount()
{
	return thread.AddWork(&TracerTI::THRDRayCount, this);
}