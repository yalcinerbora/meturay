#include "TracerBase.h"

#include "RayLib/Camera.h"
#include "RayLib/Log.h"
#include "RayLib/TracerError.h"
#include "RayLib/TracerCallbacksI.h"

#include "TracerDebug.h"
#include "GPUAcceleratorI.h"
#include "GPUMaterialI.h"
#include "TracerLogicI.h"


void TracerBase::SendError(TracerError e, bool isFatal)
{
	if(callbacks) callbacks->SendError(e);
	healthy = isFatal;
}

void TracerBase::HitRays()
{
	// Tracer Logic interface
	const Vector2i& accBitCounts = currentLogic->SceneAcceleratorMaxBits();
	GPUBaseAcceleratorI& baseAccelerator = currentLogic->BaseAcelerator();
	const AcceleratorBatchMappings& subAccelerators = currentLogic->AcceleratorBatches();

	// Reset Hit Memory for hit loop
	rayMemory.ResetHitMemory(currentRayCount, currentLogic->HitStructSize());

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

		// Reorder partitions for efficient calls
		// (sort by gpu and order for better async access)
		// ....
		// TODO:

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

void TracerBase::ShadeRays()
{
	const Vector2i matMaxBits = currentLogic->SceneMaterialMaxBits();

	// Ray Memory Pointers	
	const RayGMem* dRays = rayMemory.Rays();	
	const void* dRayAux = rayMemory.RayAux<void>();
	const HitStructPtr dHitStructs = rayMemory.HitStructs();
	const PrimitiveId* dPrimitiveIds = rayMemory.PrimitiveIds();	
	// These are sorted etc.
	HitKey* dCurrentKeys = rayMemory.CurrentKeys();
	RayId* dCurrentRayIds = rayMemory.CurrentIds();
		
	// Material Interfaces
	const MaterialBatchMappings& materials = currentLogic->MaterialBatches();
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
	rayMemory.ResizeRayOut(totalOutRayCount, currentLogic->PerRayAuxDataSize());
	unsigned char* dAuxOut = rayMemory.RayAuxOut<unsigned char>();
	RayGMem* dRaysOut = rayMemory.RaysOut();

	// Reorder partitions for efficient calls
	// (sort by gpu and order for better async access)
	// ....
	// TODO:
	
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
		void* dAuxOutStart = dAuxOut + (outOffset * currentLogic->PerRayAuxDataSize());
	
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
	: callbacks(nullptr)	
	, currentRayCount(0)
	, currentLogic(nullptr)
	, healthy(false)	
{}

void TracerBase::Initialize(int leaderGPUId)
{
	// Device initalization
	TracerError e(TracerError::END);
	if((e = CudaSystem::Initialize()) != TracerError::OK)
	{
		if(callbacks) callbacks->SendError(e);
	}
	rayMemory.SetLeaderDevice(leaderGPUId);
	CUDA_CHECK(cudaSetDevice(leaderGPUId));

	// All seems fine mark tracer as healthy
	healthy = true;
}

void TracerBase::SetOptions(const TracerOptions& opts)
{
	options = opts;
}

void TracerBase::RequestBaseAccelerator()
{}

void TracerBase::RequestAccelerator(HitKey key)
{}

void TracerBase::AttachLogic(TracerBaseLogicI& logic)
{
	// Init and set Tracer System
	TracerError e = TracerError::OK;
	if((e = logic.Initialize()) != TracerError::OK)
	{
		if(callbacks) callbacks->SendError(e);
	}
	currentLogic = &logic;

	// Initialize RNG Memory
	rngMemory = RNGMemory(logic.Seed());
}

void TracerBase::GenerateInitialRays(const GPUScene& scene,
									 int cameraId,
									 int samplePerLocation)
{
	if(!healthy) return;

	// Delegate camera ray generation to tracer system
	currentRayCount = static_cast<uint32_t>(currentLogic->GenerateRays(rayMemory, rngMemory,
																	   scene, cameraId, samplePerLocation,
																	   outputImage.Resolution(),
																	   outputImage.SegmentOffset(),
																	   outputImage.SegmentSize()));

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

void TracerBase::SetImagePixelFormat(PixelFormat f)
{
	outputImage.SetPixelFormat(f);
}

void TracerBase::ReportionImage(Vector2i start,
								Vector2i end)
{
	outputImage.Reportion(start, end);
}

void TracerBase::ResizeImage(Vector2i resolution)
{
	outputImage.Resize(resolution);
}

void TracerBase::ResetImage()
{
	outputImage.Reset();
}
