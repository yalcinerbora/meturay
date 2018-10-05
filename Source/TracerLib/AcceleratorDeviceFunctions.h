#pragma once

#include "RayStructs.h"
#include "RayLib/AABB.h"


// This is Leaf of Base Accelerator
// It points to another accelerator/material pair
struct BaseLeaf { uint32_t key; };

// Intersection function is used to determine if a leaf node data;
// which is custom, is intersects with ray. Returns valid float if such hit exists
// return NAN otherwise.
// Primitive data is custom structure that holds gpu pointers or static data globally
// Leaf struct is custom structure that holds in the leaf.
// User then can determine outcome using both.
//
// (i.e. leafStruct holds triangle index and primitive index, Primitive data holds triangles
// of multiple primitives)
//
// (i.e. leafStruct holds parameters for a sphere with necessary inverse transformation, Primitive
// data holds nothing at all)
// etc.
template <class LeafStruct, class PrimitiveData>
using IntersctionFunc = __device__ float(*)(const RayReg& r,
											const LeafStruct& data,
											const PrimitiveData& gPrimData);

// Accept hit function is used to update hit structure of the ray
// It returns immidiate termination if necessary (i.e. when any hit is enough like
// in volume rendering).
template <class HitStruct>
using AcceptHitFunc = __device__  bool(*)(HitStruct& data, RayReg& r, float newT);

// Custom bounding box generation function for primitive
template <class PrimitiveData>
using BoxGenFunc = __device__  AABB3f(*)(uint32_t primitiveId, const PrimitiveData&);

// Surface area generation function for primitive
template <class PrimitiveData>
using AreaGenFunc = __device__  float(*)(uint32_t primitiveId, const PrimitiveData&);