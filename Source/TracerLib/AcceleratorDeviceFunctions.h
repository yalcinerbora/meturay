#pragma once

#include "RayStructs.h"
#include "RayLib/AABB.h"

using HitResult = Vector<2, bool>;

// This is Leaf of Base Accelerator
// It points to another accelerator/material pair
struct BaseLeaf { uint64_t accKey; TransformId transformId; };

// Accept hit function
// Return two booleans first boolean tells control flow to terminate
// intersection checking (when finding any hit is enough), 
// other boolean is returns that the hit is accepted or not.
//
// If hit is accepted Accept hit function should return
// valid material key, primitiveId, and hit.
//
// Material Key is the index of material
// Primitive id is the index of the individual primitive
// Hit is the interpolting weights of the primitive
//
// PrimitiveData struct holds the array of the primitive data
// (normal, position etc..)
template <class Hit, class LeafStruct, class PrimitiveData>
HitResult AcceptHitFunction(// Output
							HitKey&,
							PrimitiveId&,
							Hit&,
							// I-O
							RayReg& r,
							// Input
							const LeafStruct& data,
							const PrimitiveData& gPrimData);

// Custom bounding box generation function
// For primitive
template <class PrimitiveData>
using BoxGenFunc = AABB3f(*)(PrimitiveId primitiveId, const PrimitiveData&);

// Surface area generation function for bound hierarcy generation
template <class PrimitiveData>
using AreaGenFunc = float(*)(PrimitiveId primitiveId, const PrimitiveData&);