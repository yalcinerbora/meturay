#include "VolumeGPU.cuh"

#include "RayLib/RayHitStructs.h"
#include "RayLib/Error.h"
#include "RayLib/Log.h"

VolumeGPU::VolumeGPU(const std::string& fileName, 
					 uint32_t materialId,
					 uint32_t surfaceId)
	: fileName(fileName)
	, VolumeDeviceData{Vector3ui(0u, 0u, 0u), Vector3f(0.0f, 0.0f, 0.0f), 
					   Texture3<float4>(InterpolationType::LINEAR,
										EdgeResolveType::CLAMP, false, Vector3ui(128, 128, 128)), 
					   surfaceId, materialId}
	
{}

Vector3ui VolumeGPU::Size() const
{
	return size;
}

const std::string& VolumeGPU::FileName() const
{
	return fileName;
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

NCVolumeGPU::NCVolumeGPU(const std::string& fileName,
						 uint32_t materialId,
						 uint32_t surfaceId)
	: VolumeGPU(fileName, materialId, surfaceId)
	, info{}
{}

Error NCVolumeGPU::Load()
{
	IOError e;
	if((e = MayaCache::LoadNCacheNavierStokesXML(info, fileName)) != IOError::OK)
	{
		return Error{ErrorType::IO_ERROR, static_cast<uint32_t>(e)};
	}
	size = info.dim;
	worldSize = info.size;

	// Load very first frame
	// Load Cache
	std::vector<float> velocityDensity;
	const std::string fileNameMCX = MayaCache::GenerateNCacheFrameFile(fileName, 200);
	if((e = MayaCache::LoadNCacheNavierStokes(velocityDensity, info, fileNameMCX)) != IOError::OK)
	{
		return Error{ErrorType::IO_ERROR, static_cast<uint32_t>(e)};
	}	
	// Allocate and Copy
	//mem = std::move(DeviceMemory(sizeof(Vector4f) * size[0] * size[1] * size[2]));
	//data = static_cast<Vector4*>(mem);
	volumeTex.Copy(reinterpret_cast<byte*>(velocityDensity.data()),
				   size);

	// Construct SVO
	svo.ConstructDevice(size, *this);

	//// DEBUG
	//float max = -std::numeric_limits<float>::max();
	//for(size_t i = 0; i < (size[0] * size[1] * size[2]); i++)
	//{
	//	const Vector4* dataPtr = reinterpret_cast<const Vector4*>(velocityDensity.data());
	//	max = std::max(max, dataPtr[i][3]);
	//}
	
	return Error{ErrorType::ANY_ERROR, Error::OK};
}


Error NCVolumeGPU::ChangeFrame(double time)
{
	IOError e;
	// TODO: How to determine frame from time
	uint32_t frame = static_cast<uint32_t>(time);

	METU_LOG("Loading Frame %u", frame);
	
	// Load very first frame
	// Load Cache
	std::vector<float> velocityDensity;
	const std::string fileNameMCX = MayaCache::GenerateNCacheFrameFile(fileName, frame);
	if((e = MayaCache::LoadNCacheNavierStokes(velocityDensity, info, fileNameMCX)) != IOError::OK)
	{
		return Error{ErrorType::IO_ERROR, static_cast<uint32_t>(e)};
	}
	// Allocate and Copy
	/*mem = std::move(DeviceMemory(sizeof(Vector3f) * size[0] * size[1] * size[2]));
	std::memcpy(mem, velocityDensity.data(),
				sizeof(Vector3f) * size[0] * size[1] * size[2]);*/
	volumeTex.Copy(reinterpret_cast<byte*>(velocityDensity.data()),
				   size);

	// Reconstruct SVO
	//svo.ConstructDevice(size, *this);

	return Error{ErrorType::ANY_ERROR, Error::OK};
}