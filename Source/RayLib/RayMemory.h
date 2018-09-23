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
		static constexpr uint32_t	InvalidData = 0xFFFFFFFF;

	private:
		//DeviceMemory				sortMemory;
		int							leaderDeviceId;


		DeviceMemory				memHit;
		DeviceMemory				memIn;
		DeviceMemory				memOut;

		size_t						memInMaxRayCount;
		size_t						memOutMaxRayCount;

		// Ray Related
		RayGMem*					dRayIn;
		RayGMem*					dRayOut;
		void*						dRayAuxIn;
		void*						dRayAuxOut;

		// Hit Related
		void*						dSortAuxiliary;
		HitId*						dIds0, *dIds1;		
		HitKey*						dKeys0, *dKeys1;

		//

		HitId*						dIds;
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
		RayGMem*					Rays();
		const RayGMem*				Rays() const;
		HitId*						HitIds();
		const HitId*				HitIds() const;
		HitKey*						HitKeys();
		const HitKey*				HitKeys() const;
				
		// Misc
		// Sets leader device which is responsible for sort and partition kernel calls
		void						SetLeaderDevice(int);

	

		void						Reset(size_t rayCount);

		// Memory ALlocation and reset
		void						ResizeHitMemory(size_t rayCount);
		void						ResizeRayIn(size_t rayCount, size_t perRayAuxSize);
		void						ResizeRayOut(size_t rayCount, size_t perRayAuxSize);
		void						SwapRays(size_t rayCount);

		// Common Functions
		// Sorts the hit list for multi-kernel calls
		void						SortKeys(HitId*& ids, HitKey*& keys,
											 size_t count,
											 const Vector2i& bitRange);
		// Partitions the segments for multi-kernel calls
		// Updates the ray count where the rays with 0xFF..F are considered done
		RayPartitions<uint32_t>		Partition(uint32_t& rayCount,
											  const Vector2i& bitRange);
};

inline void RayMemory::SetLeaderDevice(int deviceId)
{
	leaderDeviceId = deviceId;
}

inline RayGMem* RayMemory::Rays()
{
	return dRayIn;
}

inline const RayGMem* RayMemory::Rays() const
{
	return dRayIn;
}

inline HitId* RayMemory::HitIds()
{
	return dIds;
}

inline const HitId* RayMemory::HitIds() const
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