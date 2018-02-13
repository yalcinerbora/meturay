#include "DeviceMemory.h"

DeviceMemoryCPUBacked::DeviceMemoryCPUBacked(size_t sizeInBytes, int deviceId)
{

}

DeviceMemoryCPUBacked::DeviceMemoryCPUBacked(const DeviceMemoryCPUBacked&)
{

}

DeviceMemoryCPUBacked::DeviceMemoryCPUBacked(DeviceMemoryCPUBacked&&)
{

}

DeviceMemoryCPUBacked::~DeviceMemoryCPUBacked()
{

}

DeviceMemoryCPUBacked& DeviceMemoryCPUBacked::operator=(const DeviceMemoryCPUBacked&)
{
	return *this;
}

DeviceMemoryCPUBacked& DeviceMemoryCPUBacked::operator=(DeviceMemoryCPUBacked&&)
{
	return *this;
}

void DeviceMemoryCPUBacked::CopyToDevice(cudaStream_t stream)
{

}

void DeviceMemoryCPUBacked::CopyToHost(cudaStream_t stream)
{

}

template<class T>
constexpr T* DeviceMemoryCPUBacked::DeviceData()
{
	return reinterpret_cast<T*>(d_ptr);
}

template<class T>
constexpr const T* DeviceMemoryCPUBacked::DeviceData() const
{
	return reinterpret_cast<T*>(d_ptr);
}

template<class T>
constexpr T* DeviceMemoryCPUBacked::HostData()
{
	return reinterpret_cast<T*>(h_ptr);
}

template<class T>
constexpr const T* DeviceMemoryCPUBacked::HostData() const
{
	return reinterpret_cast<T*>(h_ptr);
}

size_t DeviceMemoryCPUBacked::Size() const
{
	return size;
}

DeviceMemory::DeviceMemory(size_t sizeInBytes)
{

}

DeviceMemory::DeviceMemory(const DeviceMemory&)
{

}

DeviceMemory::DeviceMemory(DeviceMemory&&)
{

}

DeviceMemory::~DeviceMemory()
{

}

DeviceMemory& DeviceMemory::operator=(const DeviceMemory&)
{
	return *this;
}

DeviceMemory& DeviceMemory::operator=(DeviceMemory&&)
{
	return *this;
}

template<class T>
constexpr DeviceMemory::operator T*()
{
	return reinterpret_cast<T*>(m_ptr);
}

template<class T>
constexpr DeviceMemory::operator const T*() const
{
	return reinterpret_cast<T*>(m_ptr);
}

constexpr DeviceMemory::operator void*()
{
	return m_ptr;
}

constexpr DeviceMemory::operator const void*() const
{
	return m_ptr;
}

size_t DeviceMemory::Size() const
{
	return size;
}