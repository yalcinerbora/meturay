#pragma once
/**

GPU Implementation of Mesh Related Interfaces

*/

#include "MeshI.h"
#include "DeviceMemory.h"

class MeshBatchGPU : MeshBatchI
{
	private:
		// Vertex Definiton
		std::vector<VertexElement>	elements;

		// Actual Memory
		DeviceMemory				memory;
		uint8_t*					vertexList;
		uint32_t*					baseIndex;

		uint64_t					vertexCount;
		uint64_t					indexCount;

	protected:
	public:
		// Constructors & Destructor
									MeshBatchGPU();
									MeshBatchGPU(const std::vector<std::string>& fileList);
									MeshBatchGPU(const MeshBatchGPU&);
									MeshBatchGPU(MeshBatchGPU&&) = default;
		MeshBatchGPU&				operator=(const MeshBatchGPU&);
		MeshBatchGPU&				operator=(MeshBatchGPU&&) = default;
									~MeshBatchGPU() = default;
};

/*


*/
class MeshGPU : public MeshI
{
	private:
		// Mesh unique identifier
		uint32_t objectId;

		// Offset and size, index location
		uint64_t indexStart;
		uint32_t triangleCount;

		uint64_t vertexStart;
		uint32_t vertexCount;

		// Identifiers (for referring)
		uint32_t batchId;
		uint32_t materialId;

	protected:
	public:


};