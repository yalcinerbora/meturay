#pragma once
/**

General Device memory manager for ray and it's auxiliary data


*/

#include <set>

#include "RayLib/ArrayPortion.h"
#include "RayLib/HitStructs.h"

#include "DeviceMemory.h"
#include "RayStructs.h"


template<class T>
using RayPartitions = std::set<ArrayPortion<T>>;

class RayMemory
{
	public:
		static constexpr size_t		AlignByteCount = 16;
		static constexpr int		ByteSize = 8;

	private:
		// Leader GPU device that is responsible for
		// parititoning and sorting the ray data
		// (Only usefull in multi-GPU systems)
		int							leaderDeviceId;

		// Ray Related
		DeviceMemory				memIn;
		DeviceMemory				memOut;
		// In rays are enter material kernels
		RayGMem*					dRayIn;
		// Those kernels will output one or multiple rays
		// Each material has a predefined max ray output
		// Out is allocated accordingly then materials fill it
		RayGMem*					dRayOut;
		// Each ray has auxiliary data stored in these pointers
		// Single auxiliary struct can be defined per tracer system
		// and it is common. 
		// (i.e. such struct may hold pixelId total accumulation etc)
		void*						dRayAuxIn;
		void*						dRayAuxOut;
		//---------------------------------------------------------
		// Hit Related
		// Entire Hit related memory is allocated in bulk.
		DeviceMemory				memHit;
		// MatKey holds the material group id and material group local id
		// This is used to sort rays to match kernels
		HitKey*						dMaterialKeys;
		// Transform of the hit
		// Base accelerator fill this value with a potential hit
		TransformId*				dTransformIds;
		// Primitive Id of the hit
		// Inner accelerators fill this value with a 
		// primitive group local id
		PrimitiveId*				dPrimitiveIds;
		// Custom hit Structure allocation pointer
		// This pointer is capable of holding data for all 
		// hit stuructures currently active
		// (i.e. it holds Vec2 barcy coords for triangle primitives,
		// hold position for spheres (maybe spherical coords in order to save space).
		// or other custom value for a custom primitive (spline maybe i dunno)
		HitStructPtr				dHitStructs;
		// Above code will be referenced by rayId for access
		// Since on each iteration some rays are finalzed (like out of bounds etc.)
		// We should skip them partition code will omit those rays accordingly.
		// --
		// Double buffer and temporary memory for sorting
		// Key/Index pair (key can either be accelerator or material)
		size_t						tempMemorySize;
		void*						dTempMemory;
		RayId*						dIds0, *dIds1;		
		HitKey*						dKeys0, *dKeys1;
		// Current pointers to the double buffer
		// In hit portion of the code it holds accelerator ids etc.
		HitKey*						dCurrentKeys;
		RayId*						dCurrentIds;		
		

		static void					ResizeRayMemory(RayGMem*& dRays, void*& dRayAxData,
													DeviceMemory&, 
													size_t rayCount,
													size_t perRayAuxSize);

	public:
		// Constructors & Destructor
									RayMemory();
									RayMemory(const RayMemory&) = delete;
									RayMemory(RayMemory&&) = default;
		RayMemory&					operator=(const RayMemory&) = delete;
		RayMemory&					operator=(RayMemory&&) = default;
									~RayMemory() = default;


		// Accessors
		// Ray In
		RayGMem*					Rays();		
		template<class T>
		const T*					RayAux() const;
		// Ray Out
		RayGMem*					RaysOut();
		template<class T>
		T*							RayAuxOut();

		// Hit Related		
		HitStructPtr				HitStructs();
		const HitStructPtr			HitStructs() const;
		HitKey*						MaterialKeys();
		const HitKey*				MaterialKeys() const;
		TransformId*				TransformIds();
		const TransformId*			TransformIds() const;
		PrimitiveId*				PrimitiveIds();
		const PrimitiveId*			PrimitiveIds() const;
		//		
		HitKey*						CurrentKeys();
		const HitKey*				CurrentKeys() const;
		RayId*						CurrentIds();
		const RayId*				CurrentIds() const;
		
				
		// Misc
		// Sets leader device which is responsible for sort and partition kernel calls
		void						SetLeaderDevice(int);
		int							LeaderDevice() const;
	
		// Memory Allocation
		void						ResetHitMemory(size_t rayCount, size_t hitStructSize);
		void						ResizeRayOut(size_t rayCount, size_t perRayAuxSize);
		void						SwapRays();

		// Common Functions
		// Sorts the hit list for multi-kernel calls
		void						SortKeys(RayId*& ids, HitKey*& keys,
											 size_t count,
											 const Vector2i& bitRange);
		// Partitions the segments for multi-kernel calls		
		RayPartitions<uint32_t>		Partition(uint32_t rayCount);
		// Initialize HitIds and Indices
		void						FillRayIdsForSort(uint32_t rayCount);
};

inline void RayMemory::SetLeaderDevice(int deviceId)
{
	leaderDeviceId = deviceId;
}

inline int RayMemory::LeaderDevice() const
{
	return leaderDeviceId;
}

inline RayGMem* RayMemory::Rays()
{
	return dRayIn;
}

template<class T>
inline const T* RayMemory::RayAux() const
{
	return static_cast<const T*>(dRayAuxIn);
}

inline RayGMem* RayMemory::RaysOut()
{
	return dRayOut;
}

template<class T>
inline T* RayMemory::RayAuxOut()
{
	return static_cast<T*>(dRayAuxIn);
}

inline HitStructPtr RayMemory::HitStructs()
{
	return dHitStructs;
}

inline const HitStructPtr RayMemory::HitStructs() const
{
	return dHitStructs;
}

inline HitKey* RayMemory::MaterialKeys()
{
	return dMaterialKeys;
}

inline const HitKey* RayMemory::MaterialKeys() const
{
	return dMaterialKeys;
}

inline TransformId* RayMemory::TransformIds()
{
	return dTransformIds;
}

inline const TransformId* RayMemory::TransformIds() const
{
	return dTransformIds;
}

inline PrimitiveId* RayMemory::PrimitiveIds()
{
	return dPrimitiveIds;
}

inline const PrimitiveId* RayMemory::PrimitiveIds() const
{
	return dPrimitiveIds;
}
//		
inline HitKey* RayMemory::CurrentKeys()
{
	return dCurrentKeys;
}
inline const HitKey* RayMemory::CurrentKeys() const
{
	return dCurrentKeys;
}
inline RayId* RayMemory::CurrentIds()
{
	return dCurrentIds;
}

inline const RayId* RayMemory::CurrentIds() const
{
	return dCurrentIds;
}

inline void RayMemory::ResizeRayOut(size_t rayCount, size_t perRayAuxSize)
{
	ResizeRayMemory(dRayOut, dRayAuxOut, memOut, rayCount, perRayAuxSize);
}