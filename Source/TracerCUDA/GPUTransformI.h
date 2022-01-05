#pragma once

#include <cuda.h>

#include "RayLib/Matrix.h"
#include "RayLib/SceneStructs.h"
#include "RayLib/Ray.h"
#include "RayLib/AABB.h"
#include "NodeListing.h"

class GPUTransformI;
class CudaSystem;

using GPUTransformList = std::vector<const GPUTransformI*>;

// GPU Transform Holds transformation data
// required to transform
class GPUTransformI
{
	public:
		virtual ~GPUTransformI() = default;

		// Classic Transformations
		// extra params are for non-rigid skeletal based transformations
		// or maybe morph targets
		__device__
		virtual RayF	WorldToLocal(const RayF&,
								     const uint32_t* indices = nullptr,
								     const float* weights = nullptr,
								     uint32_t count = 0) const = 0;
		__device__
		virtual Vector3	WorldToLocal(const Vector3&, bool isDirection = false,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const = 0;
		__device__
		virtual AABB3f	WorldToLocal(const AABB3f&) const = 0;

		__device__
		virtual Vector3 LocalToWorld(const Vector3&, bool isDirection = false,
									 const uint32_t* indices = nullptr,
									 const float* weights = nullptr,
									 uint32_t count = 0) const = 0;
		__device__
		virtual AABB3f	LocalToWorld(const AABB3f&) const = 0;

		__device__
		virtual QuatF	ToLocalRotation(const uint32_t* indices = nullptr,
									    const float* weights = nullptr,
									    uint32_t count = 0) const = 0;

		__device__
		virtual Vector3f ToWorldScale(const uint32_t* indices = nullptr,
									  const float* weights = nullptr,
									  uint32_t count = 0) const = 0;
		__device__
		virtual Vector3f ToLocalScale(const uint32_t* indices = nullptr,
									  const float* weights = nullptr,
									  uint32_t count = 0) const = 0;

		// Adding this function for optiX
		// Optix requires 4x3 row-major matrices for instance accelerators
		// If this transform cannot be represented by a single matrix
		// (i.e skinned transforms etc.) this function should not be called anyway
		__device__
		virtual Matrix4x4 GetLocalToWorldAsMatrix() const = 0;
};

class CPUTransformGroupI
{
	public:
		virtual								~CPUTransformGroupI() = default;

		// Interface
		virtual const char*					Type() const = 0;
		virtual const GPUTransformList&		GPUTransforms() const = 0;
		virtual SceneError					InitializeGroup(const NodeListing& transformNodes,
															double time,
															const std::string& scenePath) = 0;
		virtual SceneError					ChangeTime(const NodeListing& transformNodes, double time,
													   const std::string& scenePath) = 0;
		virtual TracerError					ConstructTransforms(const CudaSystem&) = 0;
		virtual uint32_t					TransformCount() const = 0;

		virtual size_t						UsedGPUMemory() const = 0;
		virtual size_t						UsedCPUMemory() const = 0;
};