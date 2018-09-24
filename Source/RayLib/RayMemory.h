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

		static constexpr HitKey		InvalidKey = 0xFFFFFFFF;
		static constexpr HitKey		OutsideMatKey = 0xFFFFFFFE;
		static constexpr uint32_t	InvalidData = 0xFFFFFFFF;

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
		HitGMem*					dHits;
		RayId*						dIds;
		HitKey*						dKeys;
		
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
		//RayGMem*					Rays();
		const RayGMem*				Rays() const;
		/*template<class T>
		T*							RayAux();*/
		template<class T>
		const T*					RayAux() const;

		RayGMem*					RaysOut();
		//const RayGMem*				Rays() const;
		template<class T>
		T*							RayAuxOut();
		//template<class T>
		//const T*					RayAux() const;


		// Hit Related
		HitGMem*					Hits();
		const HitGMem*				Hits() const;
		RayId*						RayIds();
		const RayId*				RayIds() const;
		HitKey*						HitKeys();
		const HitKey*				HitKeys() const;
				
		// Misc
		// Sets leader device which is responsible for sort and partition kernel calls
		void						SetLeaderDevice(int);

	

		void						Reset(size_t rayCount);

		// Memory ALlocation and reset
		void						ResetHitMemory(size_t rayCount);
		void						ResizeRayIn(size_t rayCount, size_t perRayAuxSize);
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
	return dRayIn;
}

template<class T>
inline T* RayMemory::RayAuxOut()
{
	return static_cast<T*>(dRayAuxIn);
}

inline HitGMem* RayMemory::Hits()
{
	return dHits;
}

inline const HitGMem* RayMemory::Hits() const
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