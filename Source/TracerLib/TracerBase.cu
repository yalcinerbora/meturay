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
	const Vector2i& accBitCounts = tracerSystem->SceneAcceleratorMaxBits();
	GPUBaseAcceleratorI& baseAccelerator = tracerSystem->BaseAcelerator();
	const AcceleratorBatchMappings& subAccelerators = tracerSystem->AcceleratorBatches();

	// Reset Hit Memory for hit loop
	rayMemory.ResetHitMemory(currentRayCount, tracerSystem->HitStructSize());

	// Make Base Accelerator to get ready for hitting
	baseAccelerator.GetReady(currentRayCount);

	// Ray Memory Pointers
	RayGMem* dRays = rayMemory.Rays();
	HitKey* dMaterialKeys = rayMemory.CurrentKeys();
	TransformId* dTransfomIds = rayMemory.TransformIds();
	PrimitiveId* dPrimitiveIds = rayMemory.PrimitiveIds();
	HitStructPtr dHitStructs = rayMemory.HitStructs();
	// These are sorted etc.
	HitKey* dCurrentKeys = rayMemory.CurrentKeys();	
	RayId*	dCurrentRayIds = rayMemory.CurrentIds();	
	
	// Try to hit rays until no ray is left 
	// (these rays will be assigned with a material)
	// outside rays are also assigned with a material (which is special)
	uint32_t rayCount = currentRayCount;
	while(rayCount > 0)
	{
		// Traverse accelerator
		// Base accelerator provides potential hits
		// Cannot provide an absolute hit (its not its job)
		baseAccelerator.Hit(dTransfomIds, dCurrentKeys, dRays, dCurrentRayIds,
							 rayCount);

		// Base accelerator traverses the data partially
		// Updates current key (which represents innter accelerator batch and id)
		
		// After that, system sorts rays according to the keys
		// and partitions the array according to batches

		// Sort and Partition happens on the leader device
		CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice()));

		// Sort initial results (in order to partition and launch kernels accordingly)
		// Sort is radix sort.
		// We sort inner indices in addition to batches results for better data locality
		// We only sort up-to a certain bit (radix sort) which is tied to 
		// accelerator count
		rayMemory.SortKeys(dCurrentRayIds, dCurrentKeys, rayCount, accBitCounts);
		// Parition to sub accelerators		
		//
		// There may be invalid rays sprinkled along the array.
		// Holes occur in the structure since in previous iteration,
		// a material may required to write N rays for its output (which is defined
		// by the material) but it wrote < N rays.
		// 
		// One of the main examples for such behaviour can be transparent objects
		// where ray may be only reflected (instead of refrating and reflecting) because
		// of the total internal reflection phenomena.
		auto portions = rayMemory.Partition(rayCount);

		// For each partition
		for(const auto& p : portions)
		{
			// Find Accelerator
			// Since there is no batch for invalid keys
			// that partition will be automatically be skipped
			auto loc = subAccelerators.find(p.portionId);
			if(loc == subAccelerators.end()) continue;

			RayId* dRayIdStart = dCurrentRayIds + p.offset;
			HitKey* dCurrentKeyStart = dCurrentKeys + p.offset;

			// Run local hit kernels
			// Local hit kernels returns a material key 
			// and primitive inner id.
			// Since materials are batched for both material and
			loc->second->Hit(// O
							dMaterialKeys,
							 dPrimitiveIds, 
							 dHitStructs,
							 // I-O
							 dRays,
							 // Input
							 dTransfomIds,
							 dRayIdStart, 
							 dCurrentKeyStart,
							 static_cast<uint32_t>(p.count));

			// Hit function updates material key,
			// primitive id and struct if this hit is accepted
		}

		// Update new ray count
		// On partition array check last two partitions
		// Those partitions may contain outside/invalid batches
		// Reduce ray count accordingly
		int iterationCount = std::min(static_cast<int>(portions.size()), 2);
		auto iterator = portions.rbegin();
		for(int i = 0; i < iterationCount; ++i)
		{
			const auto& portion = *iterator;
			if(portion.portionId == HitKey::NullBatch ||
			   portion.portionId == HitKey::BoundaryBatch)
			{
				rayCount = static_cast<uint32_t>(portion.offset);
			}			
			iterator++;
		}
		
		// Iteration is done
		// We cant continue loop untill these kernels are finished 
		// on gpu(s)
		//
		// Tracer logic mostly utilizies mutiple GPUs so we need to
		// wait all GPUs to finish
		CudaSystem::SyncAllGPUs();
	}
	// At the end of iteration all rays found a material, primitive
	// and interpolation weights (which should be on hitStruct)
}

void TracerBase::SendAndRecieveRays()
{
	// Here also generate RayOut and use that
	// Also pre allocate sort buffers
	// TODO:
}

void TracerBase::ShadeRays()
{
	const Vector2i matMaxBits = tracerSystem->SceneMaterialMaxBits();

	// Ray Memory Pointers	
	const RayGMem* dRays = rayMemory.Rays();	
	const void* dRayAux = rayMemory.RayAux<void>();
	const HitStructPtr dHitStructs = rayMemory.HitStructs();
	const PrimitiveId* dPrimitiveIds = rayMemory.PrimitiveIds();	
	// These are sorted etc.
	HitKey* dCurrentKeys = rayMemory.CurrentKeys();
	RayId* dCurrentRayIds = rayMemory.CurrentIds();
		
	// Material Interfaces
	const MaterialBatchMappings& materials = tracerSystem->MaterialBatches();
	uint32_t rayCount = currentRayCount;

	// Sort and Partition happens on leader device
	CUDA_CHECK(cudaSetDevice(rayMemory.LeaderDevice()));

	// Copy materialKeys to currentKeys
	// to make it ready for sorting
	rayMemory.FillRayIdsForSort(rayCount);

	// Sort with respect to the materials keys
	rayMemory.SortKeys(dCurrentRayIds, dCurrentKeys, rayCount, matMaxBits);

	// Parition w.r.t. material batch
	auto portions = rayMemory.Partition(rayCount);

	// Update new ray count
	// Last partition may be invalid partition
	// Skip those partition and adjust ray count accordingly
	if(!portions.empty() &&
	   portions.rbegin()->portionId == HitKey::NullBatch)
	{
		rayCount = static_cast<uint32_t>(portions.rbegin()->offset);
	}

	// Use partition lis to find out
	// total potential output ray count
	size_t totalOutRayCount = 0;
	for(const auto& p : portions)
	{
		auto loc = materials.find(p.portionId);
		if(loc == materials.end()) continue;

		totalOutRayCount += p.count * loc->second->OutRayCount();
	}

	// Allocate output ray memory
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
		outOffset += p.count * loc->second->OutRayCount();
		
		// Relativize input & output pointers
		const RayId* dRayIdStart = dCurrentRayIds + p.offset;
		const HitKey* dKeyStart = dCurrentKeys + p.offset;
		RayGMem* dRayOutStart = dRaysOut + outOffset;
		void* dAuxOutStart = dAuxOut + (outOffset * tracerSystem->PerRayAuxDataSize());
	
		// Actual Shade Call
		loc->second->ShadeRays(// Output
							   outputImage.GMem<Vector4f>(),
							   //
							   dRayOutStart,
							   dAuxOutStart,
							   //  Input
							   dRays,
							   dRayAux,
							   dPrimitiveIds,
							   dHitStructs,
							   //
							   dKeyStart,
							   dRayIdStart,

							   static_cast<uint32_t>(p.count),
							   rngMemory);
		
	}
	assert(totalOutRayCount == outOffset);	
	currentRayCount = static_cast<uint32_t>(totalOutRayCount);

	// Again wait all of the GPU's since
	// CUDA functions will be on multiple-gpus
	CudaSystem::SyncAllGPUs();
	
	// Shading complete
	// Now make "RayOut" to "RayIn"
	// and continue
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

void TracerBase::Initialize(TracerBaseLogicI& logic)
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
	rngMemory = RNGMemory(logic.Seed());

	// All seems fine mark tracer as healthy
	healthy = true;
}

void TracerBase::SetOptions(const TracerOptions& opts)
{
	options = opts;
}

void TracerBase::RequestBaseAccelerator()
{}

void TracerBase::RequestAccelerator(int key)
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
