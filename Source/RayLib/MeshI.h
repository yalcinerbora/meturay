#pragma once
/**

Mesh related structs

Meshes can be quite complex. (Most of the time they are similar
but per vertex data can be anything teoretically)

User should be able to define a mesh batch (a vertex definition for the object)
We consider single indexed meshes for compatability with the GPU raster APIs (oGL
DirectX etc.)

==========================================================================

Mesh batch holds actual memory of the meshes in a minimal ammount of
allocations in order to prevent memory segmentation other than that
it has no additional purpose.

For rendering etc. meshes should be used

Mesh batch stores its data struct of arrays manner and it is always
ordered depending on the logic enumeration

for N element batch data should laid out as:

P0 P1 P2 .... PN N0 N1 N2 .... NN UV0 UV1 UV2 .... UVN etc.

Then indices as 32-bit unsgined integers

==========================================================================

Mesh is a partition of the mesh batch.
It has start position and size and its triangles should be
consecutive. (Indices)

Each mesh has to have a material and can only be shaded by a single material
Each mesh has its own accelerator (i.e. BVH)

*/

#include <functional>
#include <vector>
#include <string>
#include <cstdint>

#include "AnimateI.h"
#include "SurfaceI.h"

enum class IOError;

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
	// Triangle Related
	POSITION,
	NORMAL,
	UV,
	WEIGHT,
	WEIGHT_INDEX,

	// Volume Related
	DENSITY,
	VELOCITY,

};

// Vertex Element (Logic and memory orientation of vertex (i.e. position, normal, etc)
struct VertexElement
{
	VertexElementLogic		logic;
	VertexElementDataType	dataType;
	uint8_t					componentCount;
	uint64_t				offset;
	bool					isNormalizedInt;
};

using VertexElementList = std::vector<VertexElement>;
using MeshReferences = std::vector<std::reference_wrapper<MeshI>>;

class MeshBatchI
{
	public:
		virtual								~MeshBatchI() = default;

		// Interface
		virtual const VertexElementList&	VertexElements() const = 0;

		// Accessors
		virtual uint64_t					VertexCount() const = 0;
		virtual uint64_t					IndexCount() const = 0;

		// Allocation
		virtual IOError						Load(const std::vector<std::string>& fileList) = 0;
		virtual void						Allocate(uint64_t vertexCount, 
													 uint64_t indexCount) = 0;
		
		virtual const MeshReferences&		Meshes() const = 0;
};


class MeshI : public SurfaceI
{
	public:	 
		virtual								~MeshI() = default;

		// Interface		
		// Offset Related to the batch
		virtual uint64_t					IndexStart() const = 0;
		virtual uint32_t					TriangleCount() const = 0;

		virtual uint64_t					VertexStart() const = 0;
		virtual uint32_t					VertexCount() const = 0;

		// Batch
		virtual uint32_t					BatchId() const = 0;

		// Load
		virtual IOError						LoadToMemory(const std::string& fileName,
														 uint32_t index = 0) = 0;

};

class AnimatedMeshI : public MeshI, public AnimateI
{};