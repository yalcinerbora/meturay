#pragma once
/**

Mesh related structs

Meshes can be quite complex. (Most of the time they are similar
but per vertex data can be anything teoretically)


User should be able to define a mesh batch (a vertex definition for the object)
We consider single indexed meshes for compatability with the GPU raster APIs (oGL
DirectX etc.)

*/

#include <cstdint>
#include <vector>
#include "DeviceMemory.h"

enum class VertexElementDataType : uint8_t
{
	INT8,
	INT16,
	INT32,

	UINT8,
	UINT16,
	UINT32,

	FLOAT,
	DOUBLE
};


// At the moment only these are supported for vertex
enum class VertexElementLogic : uint8_t
{
	POSITION = 0,
	NORMAL = 1,
	UV = 2,
	WEIGHT = 3,
	WEIGHT_INDEX = 4
};

struct VertexElement
{
	VertexElementLogic		logic;
	VertexElementDataType	dataType;
	uint8_t					componentCount;
	uint64_t				offset;
	bool					isNormalizedInt;
};

/**

Mesh batch holds actual memory of the meshes in a minimal ammount of
allocations in order to prevent memory segmentation other than that
it has no additional purpose.

For rendering etc. meshes should be used

Mesh batch stores its data struct of arrays manner and it is always
ordered depending on the logic enumeration

for N element batch data should laid out as:

P0 P1 P2 .... PN N0 N1 N2 .... NN UV0 UV1 UV2 .... UVN etc.

Then indices as 32-bit unsgined integers

*/
class MeshBatch
{
	private:
		// Vertex Definiton
		std::vector<VertexElement> elements;

		// Actual Memory
		DeviceMemory		memory;
		uint8_t*			vertexList;
		uint32_t*			baseIndex;

		uint32_t			vertexCount;
		uint32_t			indexCount;

	protected:

	public:
		// Constructors & Destructor
							MeshBatch();
							MeshBatch(const std::vector<std::string>& fileList);
							MeshBatch(const MeshBatch&);
							MeshBatch(MeshBatch&&) = default;
		MeshBatch&			operator=(const MeshBatch&);
		MeshBatch&			operator=(MeshBatch&&) = default;
							~MeshBatch() = default;


};

/*

Mesh is a partition of the mesh batch.
It has start position and size and its triangles should be 
consecutive. (Indices)

Each mesh has to have a material and can only be shaded by a single material
Each mesh has its own accelerator (i.e. BVH)

*/
struct Mesh
{
	// Mesh unique identifier
	uint32_t objectId;

	// Offset and size
	uint64_t startLocation;		// Index start location
	uint32_t triangleCount;
	uint32_t vertexCount;

	// Identifiers (for referring)
	uint32_t batchId;
	uint32_t materialId;
};