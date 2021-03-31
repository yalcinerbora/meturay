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

//// Converts to GPUTransform (Matrix4x4) also inverts the converted matrix
//// since we do not transform the primitive while calculating intersection
//// we transform the ray to the local space of the primitive
//__host__ __device__
//inline GPUTransform ConvnertToGPUTransform(const CPUTransform& t)
//{
//	// Device code cannot refer a global constexpr
//	// (most functions takes value by reference so)
//	constexpr Vector3 _XAxis = XAxis;
//	constexpr Vector3 _YAxis = YAxis;
//	constexpr Vector3 _ZAxis = ZAxis;
//
//	switch(t.type)
//	{
//		case TransformType::MATRIX:
//			return t.matrix.Inverse();
//		case TransformType::TRS:
//		{
//			// TRS combo is in this order;
//			// Scale, rotX, rotY, rotZ, then translate
//			Matrix4x4 result = TransformGen::Scale(t.trs.scale[0],
//												   t.trs.scale[1],
//												   t.trs.scale[2]);
//
//			result = TransformGen::Rotate(t.trs.rotation[0], _XAxis) * result;
//			result = TransformGen::Rotate(t.trs.rotation[1], _YAxis) * result;
//			result = TransformGen::Rotate(t.trs.rotation[2], _ZAxis) * result;
//			result = TransformGen::Translate(t.trs.translation);
//			return result.Inverse();
//		}
//		case TransformType::TRANSLATION:
//		{
//			return TransformGen::Translate(t.trs.translation).Inverse();
//		}
//		case TransformType::ROTATION:
//		{
//			Matrix4x4 result = TransformGen::Rotate(t.trs.rotation[0], _XAxis);
//			result = TransformGen::Rotate(t.trs.rotation[1], _YAxis) * result;
//			result = TransformGen::Rotate(t.trs.rotation[2], _ZAxis) * result;
//			return result.Inverse();
//		}
//		default: return Indentity4x4;
//	}
//}