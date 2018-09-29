#include "Debug.h"

namespace Debug
{
namespace Detail
{
	void OutputHitPairs(std::ostream& s, const RayId* ids, const HitKey* keys, size_t count);
}
}

void Debug::Detail::OutputHitPairs(std::ostream& s, const RayId* ids, const HitKey* keys, size_t count)
{
	// Do Sync this makes memory to be accessible from Host
	CUDA_CHECK(cudaDeviceSynchronize());
	for(size_t i = 0; i < count; i++)
	{
		s << "{" << std::hex << std::setw(8) << std::setfill('0') << keys[i] << ", "
				 << std::dec << std::setw(0) << std::setfill(' ') << ids[i] << "}" << " ";
	}	
}

void Debug::PrintHitPairs(const RayId* ids, const HitKey* keys, size_t count)
{
	std::stringstream s;
	Detail::OutputHitPairs(s, ids, keys, count);
	METU_LOG("%s", s.str().c_str());
}

void Debug::WriteHitPairs(const RayId* ids, const HitKey* keys, size_t count, const std::string& file)
{
	std::ofstream f(file);
	Detail::OutputHitPairs(f, ids, keys, count);
}