#include "VolumeGPU.h"
#include "RayLib/RayHitStructs.h"

VolumeGPU::VolumeGPU()
	: size(0, 0, 0)
{}

Vector3ui VolumeGPU::Size()
{
	return size;
}

IOError VolumeGPU::Load(const std::string& fileName, VolumeType t)
{
	switch(t)
	{
		case VolumeType::MAYA_NCACHE_FLUID:
			break;
		default:
			break;
	}
	return IOError::OK;
}

// From Surface Interface
uint32_t VolumeGPU::Id() const
{
	return surfaceId;
}

uint32_t VolumeGPU::MaterialId() const
{
	return materialId;
}

void VolumeGPU::HitRays(const HitRecordGMem hits,
						const ConstRayRecordGMem rays,
						uint64_t rayCount) const
{

}

void VolumeGPU::ChangeFrame(double time)
{

}