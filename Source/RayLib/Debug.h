#pragma once

#include <string>
#include <iomanip>
#include <sstream>
#include <fstream>

#include "HitStructs.h"
#include "Log.h"
#include "CudaConstants.h"

namespace Debug
{
	void PrintHitPairs(const RayId* ids, const HitKey* keys, size_t count);
	void WriteHitPairs(const RayId* ids, const HitKey* keys, size_t count, const std::string& file);

	template <class T>
	void PrintData(const T* ids, size_t count);
	template <class T>
	void WriteData(const T* ids, size_t count, const std::string& fileName);

	namespace Detail
	{
		template <class T>
		void OutputData(std::ostream& s, const T* hits, size_t count);
	}
}

template <class T>
void Debug::PrintData(const T* data, size_t count)
{
	std::stringstream s;
	Detail::OutputData(s, data, count);
	METU_LOG("%s", s.str().c_str());
}

template <class T>
void Debug::WriteData(const T* data, size_t count, const std::string& fileName)
{
	std::ofstream f(fileName);
	Detail::OutputData(f, data, count);
}

template <class T>
void Debug::Detail::OutputData(std::ostream& s, const T* data, size_t count)
{
	CUDA_CHECK(cudaDeviceSynchronize());
	for(size_t i = 0; i < count; i++)
	{
		s << std::dec << std::setw(0) << std::setfill(' ') << data[i] << ", ";
	}
}