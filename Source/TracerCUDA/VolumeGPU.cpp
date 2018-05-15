#include "VolumeGPU.h"
#include "RayLib/RayHitStructs.h"

VolumeGPU::VolumeGPU()
	: size(0, 0, 0)
{}

Vector3ui VolumeGPU::Size()
{

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
}

// From Surface Interface
uint32_t VolumeGPU::Id() const
{

}

uint32_t VolumeGPU::MaterialId() const
{

}

void VolumeGPU::HitRays(const HitRecordGMem hits,
						const ConstRayRecordGMem rays,
						uint64_t rayCount) const
{

}

void VolumeGPU::ChangeFrame(double time)
{

}