#include "Debug.h"

#include <iomanip>
#include <sstream>
#include <fstream>

#include "Log.h"
#include "CudaConstants.h"

namespace Debug
{
	void OutputHitPairs(std::ostream& s, const RayId* ids, const HitKey* keys, size_t count);
	void OutputHitGMem(std::ostream& s, const HitGMem* hits, size_t count);
	void OutputRayIds(std::ostream& s, const RayId* ids, size_t count);
}

void Debug::OutputHitPairs(std::ostream& s, const RayId* ids, const HitKey* keys, size_t count)
{
	// Do Sync this makes memory to be accessible from Host
	CUDA_CHECK(cudaDeviceSynchronize());
	for(size_t i = 0; i < count; i++)
	{
		s << "{" << std::hex << std::setw(8) << std::setfill('0') << keys[i] << ", "
				 << std::dec << std::setw(0) << std::setfill(' ') << ids[i] << "}" << " ";
	}	
}

void Debug::OutputHitGMem(std::ostream& s, const HitGMem* hits, size_t count)
{
	CUDA_CHECK(cudaDeviceSynchronize());
	for(size_t i = 0; i < count; i++)
	{
		s << i << " {" << std::hex << std::setw(8) << std::setfill('0') << hits[i].hitKey << ", "
			<< std::dec << std::setw(0) << std::setfill(' ') << hits[i].innerId << "}" << ", ";
	}
}

void Debug::OutputRayIds(std::ostream& s, const RayId* ids, size_t count)
{
	CUDA_CHECK(cudaDeviceSynchronize());
	for(size_t i = 0; i < count; i++)
	{
		s << std::dec << std::setw(0) << std::setfill(' ') << ids[i] << ", ";
	}
}

void Debug::PrintHitPairs(const RayId* ids, const HitKey* keys, size_t count)
{
	std::stringstream s;
	Debug::OutputHitPairs(s, ids, keys, count);
	METU_LOG("%s", s.str().c_str());
}

void Debug::WriteHitPairs(const RayId* ids, const HitKey* keys, size_t count, const std::string& file)
{
	std::ofstream f(file);
	Debug::OutputHitPairs(f, ids, keys, count);
}

void Debug::PrintHitGMem(const HitGMem* hits, size_t count)
{
	std::stringstream s;
	Debug::OutputHitGMem(s, hits, count);
	METU_LOG("%s", s.str().c_str());
}

void Debug::WriteHitGMem(const HitGMem* hits, size_t count, const std::string& fileName)
{
	std::ofstream f(fileName);
	Debug::OutputHitGMem(f, hits, count);
}

void Debug::PrintRayIds(const RayId* ids, size_t count)
{
	std::stringstream s;
	Debug::OutputRayIds(s, ids, count);
	METU_LOG("%s", s.str().c_str());
}

void Debug::WriteRayIds(const RayId* ids, size_t count, const std::string& fileName)
{
	std::ofstream f(fileName);
	Debug::OutputRayIds(f, ids, count);
}