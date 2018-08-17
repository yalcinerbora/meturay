#include "SVODevice.cuh"
#include "VolumeGPU.cuh"
#include "RayLib/CudaConstants.h"

__global__
void KCConstruct(SVODeviceData svo,
				 uint32_t* allocator,
				 uint32_t allocatorMax,
				 const VolumeDeviceData volume)
{
	// Kernel Grid-Stride Loop
	for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
		threadId < volume.LinearSize();
		threadId += (blockDim.x * gridDim.x))
	{

		//const unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
		Vector3i voxelId(static_cast<int>(threadId / (volume.Size()[0] * volume.Size()[1])),
						 static_cast<int>((threadId / volume.Size()[0]) % volume.Size()[1]),
						 static_cast<int>(threadId % volume.Size()[0]));

		if(voxelId[0] >= 128 ||
		   voxelId[1] >= 128 ||
		   voxelId[2] >= 128)
		{
			assert(false);
		}

		bool needsLeaf = false;
		UNROLL_LOOP			
		for(int i = 0; i < 8; i++)
		{
			Vector3i offset((i >> 0) & 0x1,
							(i >> 1) & 0x1,
							(i >> 2) & 0x1);
			offset = -offset;

			Vector3i index = voxelId + offset;
			if(index[0] < 0 ||
			   index[1] < 1 ||
			   index[2] < 2) continue;
			else 
				needsLeaf |= volume.HasData(static_cast<Vector3ui>(index));
		}
		if(!needsLeaf) continue;

		// Iterate untill level (This portion's nodes should be allocated)
		SVODeviceData::Node* node = svo.d_root;
		for(uint32_t i = 1; i <= svo.totalLevel; i++)
		{
			uint32_t allocNode = svo.AtomicAllocateNode(node, allocator);
			uint32_t childId = SVODeviceData::CalculateLevelChildId(voxelId, i, svo.totalLevel);
			assert(childId < 8);

			// Exit if allocation is full
			if(allocNode + childId >= allocatorMax) return;

			node = svo.d_root + allocNode + childId;
		}
		// Node found mark it
		node->next = SVODeviceData::DATA_LEAF;
	}
}

__device__
inline uint32_t SVODeviceData::AtomicAllocateNode(Node* gNode, 
												  uint32_t* gLevelAllocator)
{
	// Release Configuration Optimization fucks up the code
	// Prob changes some memory read/write ordering change
	// Its fixed but comment is here for future
	// Problem here was cople threads on the same warp waits eachother and
	// after some memory ordering changes by compiler responsible thread waits
	// other threads execution to be done
	// Code becomes something like this after compiler changes some memory orderings
	//
	//	while(old = atomicCAS(gNode, 0xFFFFFFFF, 0xFFFFFFFE) == 0xFFFFFFFE); <-- notice semicolon
	//	 if(old == 0xFFFFFF)
	//		location = allocate();
	//	location = old;
	//	return location;
	//
	// first allocating thread will never return from that loop since 
	// its warp threads are on infinite loop (so deadlock)
	//
	// much cooler version can be warp level exchange intrinsics
	// which slightly reduces atomic pressure on the single node (on lower tree levels atleast)
	//
	// 0xFFFFFFFF means empty (non-allocated) node
	// 0xFFFFFFFE means allocation in progress
	// All other numbers are valid nodes (unless of course those are out of bounds)

	// Just take node if already allocated
	if(gNode->next < DATA_LEAF) return gNode->next;
	// Try to lock the node and allocate for that node
	uint32_t old = ALLOCATION_IN_PROGRESS;
	while(old == ALLOCATION_IN_PROGRESS)
	{
		old = atomicCAS(&gNode->next, NULL_CHILD, ALLOCATION_IN_PROGRESS);
		if(old == NULL_CHILD)
		{
			// Allocate
			uint32_t location = atomicAdd(gLevelAllocator, 8);
			reinterpret_cast<volatile uint32_t&>(gNode->next) = location;
			old = location;
		}
		__threadfence();	// This is important somehow compiler changes this and makes infinite loop on same warp threads
	}
	return old;
}

void SVODevice::IncreaseMemory(uint32_t nodeCount)
{
	// Space for allocator (Allocate for single node for proper alignment
	size_t totalSize = (nodeCount + 1) * sizeof(Node);
	size_t oldSize = (d_allocator != nullptr) ? (*d_allocator + 1) * sizeof(Node) : 0;

	DeviceMemory memNew = DeviceMemory(totalSize);
	std::memset(memNew, 0xFFFFFFFF, totalSize);
	*static_cast<uint32_t*>(memNew) = 1;
	if(oldSize != 0)
	{
		CUDA_CHECK(cudaMemcpy(memNew, mem, oldSize, cudaMemcpyDeviceToDevice));		
	}
	mem = std::move(memNew);
	d_allocator = static_cast<uint32_t*>(mem);
	d_root = static_cast<Node*>(mem) + 1;

	totalNodeCount = nodeCount;
}

SVODevice::SVODevice()
	: SVODeviceData{Zero3, Zero3, Zero3ui, 0, 0, nullptr}
	, d_allocator(nullptr)
	, totalNodeCount(0)
{}

void SVODevice::ConstructDevice(const Vector3ui& volumeSize, 
								const VolumeI& v)
{
	const VolumeDeviceData& volume = static_cast<const VolumeGPU&>(v);

	// Change This
	
	aabbMin = Vector3(-5.0f, 0.0f, -5.0f);
	aabbMax = aabbMin + volume.worldSize;
	dimensions = volume.size;	
	totalLevel = static_cast<uint32_t>(std::log2(std::max(std::max(volume.size[0], volume.size[1]), volume.size[2])));
	volumeId = volume.surfaceId;	
	return;

	//uint32_t nodeCount = totalNodeCount;
	//if(nodeCount == 0)
	//{
	//	nodeCount = InitialNodeCount;
	//	IncreaseMemory(nodeCount);
	//}
	//else
	//{
	//	CUDA_CHECK(cudaMemset(d_root, 0xFF, totalNodeCount * sizeof(Node)));
	//	CUDA_CHECK(cudaMemset(d_allocator, 0x00, 1 * sizeof(Node)));
	//}
	//do
	//{		
	//	// Construction Call SVO
	//	CudaSystem::GPUCallX(CudaSystem::CURRENT_DEVICE, 
	//						 0, 0, KCConstruct,
	//						 *this, d_allocator,
	//						 totalNodeCount, volume);		
	//	CUDA_CHECK(cudaDeviceSynchronize());

	//	//
	//	if(*d_allocator + 7 >= nodeCount)
	//	{
	//		// We need to reallocate
	//		nodeCount = static_cast<uint32_t>(nodeCount * NodeIncrementRatio);
	//		IncreaseMemory(nodeCount);
	//		continue;
	//	}
	//	else break;
	//} while(true);
}

size_t SVODevice::TotalNodeSize()
{
	return totalNodeCount;
}