#pragma once

#include "RayLib/VolumeI.h"
#include "RayLib/DeviceMemory.h"
#include "RayLib/Vector.h"

class VolumeGPU : VolumeI
{
	private:
		DeviceMemory		mem;
		
		Vector3ui			size;

		uint32_t			surfaceId;
		uint32_t			materialId;

	protected:
	public:
		// Constructors & Destructor
							VolumeGPU();
							VolumeGPU(const VolumeGPU&) = default;
							VolumeGPU(VolumeGPU&&) = default;
		VolumeGPU&			operator=(const VolumeGPU&) = default;
		VolumeGPU&			operator=(VolumeGPU&&) = default;
							~VolumeGPU() = default;
							
		//
		Vector3ui			Size() override;
		IOError				Load(const std::string& fileName, VolumeType) override;
		
		// From Surface Interface
		uint32_t			Id() const override;
		uint32_t			MaterialId() const override;
		void				HitRays(const HitRecordGMem,
									const ConstRayRecordGMem,
									uint64_t rayCount) const override;

		// From Animate Interface
		void				ChangeFrame(double time) override;

		//
		__device__ bool		HasData(const Vector3ui&) const;
};

__device__ 
inline bool VolumeGPU::HasData(const Vector3ui& index) const
{

}