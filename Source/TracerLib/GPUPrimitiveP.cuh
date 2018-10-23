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

template <class T, template <class> class... Ps>
constexpr bool satisfies_all_v = std::conjunction<Ps<T>...>::value;

template <class HitData, class PrimitiveData, class LeafData,
		  AcceptHitFunction<HitData, PrimitiveData, LeafData> HitFunc,
		  LeafGenFunction<PrimitiveData, LeafData> LeafFunc,
		  BoxGenFunction<PrimitiveData> BoxFunc,
		  AreaGenFunction<PrimitiveData> AreaFunc>
class GPUPrimitiveGroup : public GPUPrimitiveGroupI
{
	public:	
	   	// Type Definitions for kernel generations
		using PrimitiveData						= PrimitiveData;
		using HitData							= HitData;
		using LeafStruct						= LeafStruct;
		// Function Definitions
		// Used by accelerator definitions etc.
		static constexpr auto HitFunc			= HitFunc;
		static constexpr auto LeafFunc			= LeafFunc;		
		static constexpr auto BoxFunc			= BoxFunc;
		static constexpr auto AreaFunc			= AreaFunc;
		
	private:
	protected:
		PrimitiveData					dData = PrimitiveData{};

	public:
		// Constructors & Destructor
										GPUPrimitiveGroup() = default;
		virtual							~GPUPrimitiveGroup() = default;
};