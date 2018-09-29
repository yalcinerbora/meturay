#pragma once
/**

General Device memory manager for ray and it's auxiliary data


*/

#include "DeviceMemory.h"
#include "RayStructs.h"
#include "HitStructs.h"

#include <cstdint>
#include <cassert>
#include <set>

#include "RayLib/ArrayPortion.h"

template<class T>
using RayPartitions = std::set<ArrayPortion<T>>;

class RayMemory
{
	public:
		static constexpr size_t		AlignByteCount = 16;
		static constexpr int		ByteSize = 8;

	private:
		int							leaderDeviceId;

		DeviceMemory				memHit;
		DeviceMemory				memIn;
		DeviceMemory				memOut;

		// Ray Related
		RayGMem*					dRayIn;
		RayGMem*					dRayOut;
		void*						dRayAuxIn;
		void*						dRayAuxOut;
	
		// Hit Related
		size_t						tempMemorySize;
		void*						dTempMemory;
		RayId*						dIds0, *dIds1;		
		HitKey*						dKeys0, *dKeys1;
		//
		HitKey*						dCurrentHits;		
		RayId*						dIds;		
		//
		HitKey*						dPotentialHits;
		void*						dHitStructs;

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
		const RayGMem*				Rays() const;		
		template<class T>
		const T*					RayAux() const;
		// Ray Out
		RayGMem*					RaysOut();
		template<class T>
		T*							RayAuxOut();

		// Hit Related
		void*						HitStructs();
		void*						HitStructs() const;
		HitKey*						CurrentHits();
		const HitKey*				CurrentHits() const;
		//
		HitKey*						PotentialHits();
		const HitKey*				PotentialHits() const;
		RayId*						RayIds();
		const RayId*				RayIds() const;
		
				
		// Misc
		// Sets leader device which is responsible for sort and partition kernel calls
		void						SetLeaderDevice(int);
		int							LeaderDevice() const;
	
		// Memory Allocation
		void						ResetHitMemory(size_t rayCount, size_t hitStructSize);
//		void						ResizeRayIn(size_t rayCount, size_t perRayAuxSize);
		void						ResizeRayOut(size_t rayCount, size_t perRayAuxSize);
		void						SwapRays();

		// Common Functions
		// Sorts the hit list for multi-kernel calls
		void						SortKeys(RayId*& ids, HitKey*& keys,
											 size_t count,
											 const Vector2i& bitRange);
		// Partitions the segments for multi-kernel calls
		// Updates the ray count where the rays with 0xFF..F are considered done
		RayPartitions<uint32_t>		Partition(uint32_t& rayCount,
											  const Vector2i& bitRange);
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

inline const RayGMem* RayMemory::Rays() const
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

inline HitKey* RayMemory::HitFinals()
{
	return dHits;
}

inline const HitKey* RayMemory::HitFinals() const
{
	return dHits;
}

inline RayId* RayMemory::RayIds()
{
	return dIds;
}

inline const RayId* RayMemory::RayIds() const
{
	return dIds;
}

inline HitKey* RayMemory::HitKeys()
{
	return dKeys;
}

inline const HitKey* RayMemory::HitKeys() const
{
	return dKeys;
}

//inline void RayMemory::ResizeRayIn(size_t rayCount, size_t perRayAuxSize)
//{
//	ResizeRayMemory(dRayIn, dRayAuxIn, memIn, rayCount, perRayAuxSize);
//}

inline void RayMemory::ResizeRayOut(size_t rayCount, size_t perRayAuxSize)
{
	ResizeRayMemory(dRayOut, dRayAuxOut, memOut, rayCount, perRayAuxSize);
}