#pragma once

#include "HitStructs.cuh"

// Default Leaf Struct
// Most of the leaf structs will have these
// data but still it is user defined.
struct DefaultLeaf
{
	PrimitiveId		primitiveId;
	HitKey			matId;
};

struct EmptyLeaf {};


template <class PrimData>
__device__ __host__
inline EmptyLeaf GenerateEmptyLeaf(const HitKey matId,
								   const PrimitiveId primitiveId,
								   const PrimData& primData)
{
	return {};
}

template <class PrimData>
__device__ __host__
inline DefaultLeaf GenerateDefaultLeaf(const HitKey matId,
									   const PrimitiveId primitiveId,
									   const PrimData& primData)
{
	return DefaultLeaf{primitiveId, matId};
}