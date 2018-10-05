#include "TracerBase.h"

#include "RayLib/Camera.h"
#include "RayLib/Log.h"
#include "RayLib/TracerError.h"

#include "TracerDebug.h"
#include "GPUAcceleratorI.h"
#include "GPUMaterialI.h"
#include "TracerLogicI.h"

void TracerBase::SendError(TracerError e, bool isFatal)
{
	if(errorFunc) errorFunc(e);
	healthy = isFatal;
}

void TracerBase::HitRays()
{
	// Tracer Logic interface
	const Vector2i accBitRange = tracerSystem->AcceleratorBitRange();
	const GPUBaseAcceleratorI* baseAccelerator = tracerSystem->BaseAcelerator();
	const AcceleratorGroupMappings& subAccelerators = tracerSystem->AcceleratorGroups();

	// Reset Hit Memory for hit loop
	rayMemory.ResetHitMemory(currentRayCount, tracerSystem->HitStructMaxSize());

	// Ray Memory Pointers
	RayGMem* dRays = rayMemory.Rays();	
	HitKey* dCurrentHits = rayMemory.CurrentHits();
	void* dHitStructs = rayMemory.HitStructs<unsigned char>();
	// These are sorted etc.
	HitKey* dPotentialHits = rayMemory.PotentialHits();	
	RayId*	dRayIds = rayMemory.RayIds();	
	
	// Try to hit rays until no ray is left 
	// (these rays will be assigned with a material)
	// outside rays are also assigned with a material (which is special)
	uint32_t rayCount = currentRayCount;
	while(rayCount > 0)
	{
		// Traverse accelerator
		// Base accelerator provides potential hits
		// Cannot provide an absolute hit (its not its job)
		baseAccelerator->Hit(dPotentialHits, dRays, dRayIds,
							 rayCount);

		// Base accelerator traverses the data partially
		// It delegates the rays to smaller accelerators
		// by writing their Id's to its portion in the key.

		// After that systems sorts ray hit list and key
		// and partitions the array this partitioning scheme 

		// Sort and Partition happens on the leader device
		CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice()));

		// Sort initial results (to partition and launch kernels accordingly)				
		rayMemory.SortKeys(dRayIds, dPotentialHits, rayCount, accBitRange);
		// Parition to sub accelerators
		// Remove the rays that are invalid.
		//
		// Partition code does not return invalid rays.
		// Invalid rays include empty rays (which are holes in the array)
		// or missed rays (that does not hit anything).
		// If accelerator bit segment is used for partitioning,
		// portions structure omits both of these type of rays.
		//
		// Holes occur in the structure since in previous iteration,
		// a material may required to write N rays for its output (which is defined
		// by the material) but it wrote < N rays.
		// 
		// One of the main examples for such behaviour can be transparent objects
		// where ray may be only reflected (instead of refrating and reflecting) because
		// of the total internal reflection phenomena.
		auto portions = rayMemory.Partition(rayCount, accBitRange);

		// For each partition
		for(const auto& p : portions)
		{
			auto loc = subAccelerators.find(p.portionId);
			if(loc == subAccelerators.end()) continue;

			// Run local hit kernels
			// These hit kernels can only modify actual hits
			// Potential HitKeys are used to fetch inner data
			RayId* dRayIdStart = dRayIds + p.offset;
			HitKey* dPotentialHitStart = dPotentialHits + p.offset;

			loc->second->Hit(dRays, dHitStructs, dCurrentHits,
							 dRayIdStart, dPotentialHitStart,
							 static_cast<uint32_t>(p.count));

			// Hit function updates the hitIds structure with its appropirate data,
			// internally, also it changes HitId structure if new hit is found

		}
		// Iteration is done
		// We cant continue loop untill these kernels are finished 
		// on gpu(s)
		CUDA_CHECK(cudaDeviceSynchronize());
	}
	// At the end of iteration each accelerator holds its custom struct array
	// And hit ids holds a index for that struct
}

void TracerBase::SendAndRecieveRays()
{
	// Here also generate RayOut and use that
	// Also pre allocate sort buffers
	// TODO:
}

void TracerBase::ShadeRays()
{
	const Vector2i matBitRange = tracerSystem->MaterialBitRange();

	// Ray Memory Pointers	
	const RayGMem* dRays = rayMemory.Rays();	
	HitKey* dPotentialHits = rayMemory.PotentialHits();
	const void* dHitStructs = rayMemory.HitStructs<void>();
	RayId*	dRayIds = rayMemory.RayIds();
	const void* dAux = rayMemory.RayAux<void>();
	
	// Material Interfaces
	const MaterialGroupMappings& materials = tracerSystem->MaterialGroups();
		
	// Now here conside incoming rays from different tracers
	// Consume ray array
	uint32_t rayCount = currentRayCount;

	// Sort and Partition happens on leader device
	CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice()));

	// Copy Keys (which are stored in HitGMem) to HitKeys
	// Make ready for sorting
	rayMemory.FillRayIdsForSort(rayCount);

	// Sort with respect to the hits that are returned
	rayMemory.SortKeys(dRayIds, dPotentialHits, rayCount, matBitRange);

	// Parition w.r.t. material (full range sort is required here)
	// Each same material on accelerator is actually considered a unique material
	// this is why we sort full range.
	// This is required since same material may fetch is data differently 
	// from different objects
	auto portions = rayMemory.Partition(rayCount, matBitRange);

	// Use partition lis to find out
	// total potential output ray count
	size_t totalOutRayCount = 0;
	for(const auto& p : portions)
	{
		auto loc = materials.find(p.portionId);
		if(loc == materials.end()) continue;

		totalOutRayCount += p.count * loc->second->MaxOutRayPerRay();
	}

	// Allocate
	rayMemory.ResizeRayOut(totalOutRayCount, tracerSystem->PerRayAuxDataSize());
	unsigned char* dAuxOut = rayMemory.RayAuxOut<unsigned char>();
	RayGMem* dRaysOut = rayMemory.RaysOut();

	// For each partition
	size_t outOffset = 0;
	for(const auto& p : portions)
	{
		auto loc = materials.find(p.portionId);
		if(loc == materials.end()) continue;

		// Since output is dynamic (each material may write multiple rays)
		// add offsets to find proper count
		outOffset += p.count * loc->second->MaxOutRayPerRay();
		
		// Run local hit kernels
		RayId* dRayIdStart = dRayIds + p.offset;
		RayGMem* dRayOutStart = dRaysOut + outOffset;
		void* dAuxOutStart = dAuxOut + (outOffset * tracerSystem->PerRayAuxDataSize());
	
		// Actual Shade Call
		// TODO: Defer this call if p.count is too low
		// Problem: What if it is always low ?
		// Probably it is better to launch it
		//
		// Another TODO: Implement multi-gpu load balancing
		// More TODO: Implement single-gpu SM load balacing
		loc->second->ShadeRays(dRayOutStart, dAuxOutStart,
							   dRays, dHitStructs, dAux,
							   dRayIdStart,
							   static_cast<uint32_t>(p.count),
							   rngMemory);
		
	}
	assert(totalOutRayCount == outOffset);	
	currentRayCount = static_cast<uint32_t>(totalOutRayCount);

	// Shading complete
	// Now make "RayOut" to "RayIn"
	rayMemory.SwapRays();
}

TracerBase::TracerBase()
	: rayDelegateFunc(nullptr)
	, errorFunc(nullptr)
	, analyticFunc(nullptr)
	, imageFunc(nullptr)
	, baseSendFunc(nullptr)
	, accSendFunc(nullptr)
	, currentRayCount(0)
	, tracerSystem(nullptr)
	, healthy(false)	
{}

void TracerBase::Initialize(uint32_t seed, TracerLogicI& logic)
{
	// Device initalization
	TracerError e(TracerError::END);
	if((e = CudaSystem::Initialize()) != TracerError::OK)
	{
		if(errorFunc) errorFunc(e);
	}

	// Init and set Tracer System
	if((e = logic.Initialize()) != TracerError::OK)
	{
		if(errorFunc) errorFunc(e);
	}
	tracerSystem = &logic;

	// Select a leader device that is responsible
	// for sorting and partitioning works
	// for different materials / accelerators
	// TODO: Determine a leader Device
	rayMemory.SetLeaderDevice(0);
	CUDA_CHECK(cudaSetDevice(0));

	// Initialize RNG Memory
	rngMemory = RNGMemory(seed);

	// All seems fine mark tracer as healthy
	healthy = true;
}

void TracerBase::SetTime(double seconds)
{}

void TracerBase::SetParams(const TracerParameters&)
{}

void TracerBase::SetScene(const std::string& sceneFileName)
{}

void TracerBase::RequestBaseAccelerator()
{}

void TracerBase::RequestAccelerator(HitKey key)
{}

void TracerBase::AssignAllMaterials()
{}

void TracerBase::AssignMaterial(uint32_t matId)
{}

void TracerBase::UnassignAllMaterials()
{}

void TracerBase::UnassignMaterial(uint32_t matId)
{}

void TracerBase::GenerateCameraRays(const CameraPerspective& camera,
									const uint32_t samplePerPixel)
{
	if(!healthy) return;

	// Initial ray count
	currentRayCount = outputImage.SegmentSize()[0] *
					  outputImage.SegmentSize()[1] *
					  samplePerPixel * samplePerPixel;

	// Allocate enough space for ray
	rayMemory.ResizeRayOut(currentRayCount, tracerSystem->PerRayAuxDataSize());

	// Delegate camera ray generation to tracer system
	tracerSystem->GenerateCameraRays(rayMemory, rngMemory,
									 camera, samplePerPixel,
									 outputImage.Resolution(),
									 outputImage.SegmentOffset(),
									 outputImage.SegmentSize());

	

	// You can only write to out buffer of the ray memory
	// Make that memory in rays for hit/shade system
	rayMemory.SwapRays();
}

bool TracerBase::Continue()
{
	return (currentRayCount > 0) && healthy;
}

void TracerBase::Render()
{
	if(!healthy) return;
	if(currentRayCount == 0) return;
	
	HitRays();
	ShadeRays();

	METU_LOG("-----------------------------");
	METU_LOG("-----------------------------");
}

void TracerBase::FinishSamples()
{
	if(!healthy) return;
}

bool TracerBase::IsCrashed()
{
	return (!healthy);
}

void TracerBase::AddMaterialRays(const RayCPU&, const HitCPU&,
								 uint32_t rayCount, uint32_t matId)
{}

void TracerBase::SetImagePixelFormat(PixelFormat f)
{
	outputImage.SetPixelFormat(f);
}

void TracerBase::ReportionImage(const Vector2ui& offset,
								const Vector2ui& size)
{
	outputImage.Reportion(offset, size);
}

void TracerBase::ResizeImage(const Vector2ui& resolution)
{
	outputImage.Resize(resolution);
}

void TracerBase::ResetImage()
{
	outputImage.Reset();
}
