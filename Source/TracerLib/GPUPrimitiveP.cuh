#pragma once
/**

Template Wrapper for Primitives
P tag on name is for partial implementation
to guide the primitive implementor to make creations 
proper for combined templates

*/

#include "RayLib/SceneStructs.h"

#include "GPUPrimitiveI.h"
#include "AcceleratorDeviceFunctions.h"

template<class T, std::enable_if_t<std::is_function_v<T::AcceptFunc> &&
				  std::is_function_v<T::GenLeafFunc> &&
				  (sizeof(T) > 0)>>
struct is_valid_primitive_group//<T>
	: std::true_type
{};


template <class T, template <class> class... Ps>
constexpr bool satisfies_all_v = std::conjunction<Ps<T>...>::value;

//template <class PrimitiveData, class HitStruct, class LeafStruct,
//Func, Func, Func>
//class GPUPrimitiveGroup : public GPUPrimitiveGroupI
//{
//	public:	
//	   	// Type Definitions for kernel generations
//		using PrimitiveData						= PrimitiveData;
//		using HitReg							= HitStruct;
//		using LeafStruct						= LeafStruct;
//		static constexpr auto AcceptFunc		= TriangleClosestHit;
//		static constexpr auto GenLeafFunc		= GenerateLeaf<PrimitiveData>;
//		// 
//		static constexpr auto AABBGenFunc		= TriangleClosestHit;
//		static constexpr auto AreaGenFunc		= TriangleClosestHit;
//		
//	private:
//		__device__ __host__
//		static HitResult TriangleClosestHit(// Output
//											HitKey& newMat,
//											PrimitiveId& newPrimitive,
//											TriangleHit& newHit,
//											// I-O
//											RayReg& rayData,
//											// Input
//											const TriData& primData,
//											const DefaultLeaf& leaf);
//
//		__device__ __host__
//		static LeafStruct GenerateLeaf(const HitKey matId,
//									   const PrimitiveId primitiveId,
//									   const PrimData& primData);
//
//		stat
//		__device__ __host__
//
//	protected:
//		PrimitiveData					dData = PrimitiveData{};
//
//	public:
//										GPUPrimitiveGroup() = default;
//										~GPUPrimitiveGroup() = default;
//};