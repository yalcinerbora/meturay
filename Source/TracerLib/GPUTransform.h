#pragma once

#include <cuda.h>

#include "RayLib/Matrix.h"
#include "RayLib/SceneStructs.h"

static constexpr uint32_t DEFAULT_TRANSFORM_INDEX = 0;

using GPUTransform = Matrix4x4;

class GPUTransfomI
{
	public:
		virtual ~GPUTransfomI() = default;


		Matrix4x4 Transform(uint32_t index) = 0;

		//
		virtual RayF WorldToLocal(const RayF& r) = 0;

		template<class Prim>
		virtual RayF WorldToTangent(const RayF&,
									const typename Prim::Data& d,
									PrimitiveId&) = 0;


};


// Converts to GPUTransform (Matrix4x4) also inverts the converted matrix
// since we do not transform the primitive while calculating intersection
// we transform the ray to the local space of the primitive
__host__ __device__ 
inline GPUTransform ConvnertToGPUTransform(const CPUTransform& t)
{
	// Device code cannot refer a global constexpr 
	// (most functions takes value by reference so)
	constexpr Vector3 _XAxis = XAxis;
	constexpr Vector3 _YAxis = YAxis;
	constexpr Vector3 _ZAxis = ZAxis;

	switch(t.type)
	{
		case TransformType::MATRIX:
			return t.matrix.Inverse();
		case TransformType::TRS:
		{
			// TRS combo is in this order;
			// Scale, rotX, rotY, rotZ, then translate
			Matrix4x4 result = TransformGen::Scale(t.trs.scale[0],
												  t.trs.scale[1],
												  t.trs.scale[2]);

			result = TransformGen::Rotate(t.trs.rotation[0], _XAxis) * result;
			result = TransformGen::Rotate(t.trs.rotation[1], _YAxis) * result;
			result = TransformGen::Rotate(t.trs.rotation[2], _ZAxis) * result;
			result = TransformGen::Translate(t.trs.translation);
			return result.Inverse();
		}
		case TransformType::TRANSLATION:
		{
			return TransformGen::Translate(t.trs.translation).Inverse();
		}
		case TransformType::ROTATION:
		{
			Matrix4x4 result = TransformGen::Rotate(t.trs.rotation[0], _XAxis);
			result = TransformGen::Rotate(t.trs.rotation[1], _YAxis) * result;
			result = TransformGen::Rotate(t.trs.rotation[2], _ZAxis) * result;
			return result.Inverse();
		}
		default: return Indentity4x4;
	}
}