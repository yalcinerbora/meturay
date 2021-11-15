#pragma once

/**

CUDA Device Memory RAII principle classes

New unified memory classes are used where applicable
These are wrapper of cuda functions and their most important responsiblity is
to delete allocated memory

All of the operations (execpt allocation) are asyncronious.

TODO: should we interface these?

*/
#include <cuda_runtime.h>

#include <limits>
#include <fstream>
#include <iostream>
#include <tuple>
#include <cassert>

#include "RayLib/MemoryAlignment.h"
#include "RayLib/Types.h"

class DeviceMemoryI
{
    public:
        virtual             ~DeviceMemoryI() = default;

        // Interface
        virtual size_t      Size() const = 0;

};

// Basic semi-interface for memories that are static for each GPU
// Textures are one example
class DeviceLocalMemoryI : public DeviceMemoryI
{
    private:
    protected:
        int                     currentDevice;

    public:
                                DeviceLocalMemoryI(int initalDevice = 0) : currentDevice(initalDevice) {}
        virtual                 ~DeviceLocalMemoryI() = default;

        // Interface
        virtual void            MigrateToOtherDevice(int deviceTo, cudaStream_t stream = (cudaStream_t)0) = 0;
};

// Has a CPU Image of current memory
// Usefull for device static memory that can be generated at CPU while
// GPU doing work on GPU memory
// in our case some form of function backed animation can be calculated using these)
class DeviceMemoryCPUBacked : public DeviceLocalMemoryI
{
    private:
        void*                       h_ptr;
        void*                       d_ptr;

        size_t                      size;

    protected:
    public:
        // Constructors & Destructor
                                    DeviceMemoryCPUBacked();
                                    DeviceMemoryCPUBacked(size_t sizeInBytes, int deviceId = 0);
                                    DeviceMemoryCPUBacked(const DeviceMemoryCPUBacked&);
                                    DeviceMemoryCPUBacked(DeviceMemoryCPUBacked&&);
                                    ~DeviceMemoryCPUBacked();
        DeviceMemoryCPUBacked&      operator=(const DeviceMemoryCPUBacked&);
        DeviceMemoryCPUBacked&      operator=(DeviceMemoryCPUBacked&&);

        // Memcopy
        void                        CopyToDevice(size_t offset = 0, size_t copySize = std::numeric_limits<size_t>::max(), cudaStream_t stream = (cudaStream_t)0);
        void                        CopyToHost(size_t offset = 0, size_t copySize = std::numeric_limits<size_t>::max(), cudaStream_t stream = (cudaStream_t)0);

        // Access
        template<class T>
        constexpr T*                DeviceData();
        template<class T>
        constexpr const T*          DeviceData() const;
        template<class T>
        constexpr T*                HostData();
        template<class T>
        constexpr const T*          HostData() const;
        // Misc
        size_t                      Size() const override;
        // Interface
        void                        MigrateToOtherDevice(int deviceTo, cudaStream_t stream = (cudaStream_t)0) override;
};

// Generic Device Memory (most of the cases this should be used)
// Fire and forget type memory
// In our case rays and hit records will be stored in this form
class DeviceMemory : public DeviceMemoryI
{
    private:
        void*                       m_ptr;  // managed pointer

        size_t                      size;

    protected:
    public:
        // Constructors & Destructor
                                    DeviceMemory();
                                    DeviceMemory(size_t sizeInBytes);
                                    DeviceMemory(const DeviceMemory&);
                                    DeviceMemory(DeviceMemory&&);
                                    ~DeviceMemory();
        DeviceMemory&               operator=(const DeviceMemory&);
        DeviceMemory&               operator=(DeviceMemory&&);

        // Access
        template<class T>
        constexpr explicit          operator T*();
        template<class T>
        constexpr explicit          operator const T*() const;
        constexpr                   operator void*();
        constexpr                   operator const void*() const;

        // Misc
        size_t                      Size() const override;

        static void                 EnlargeBuffer(DeviceMemory&, size_t);

        template <class... Args>
        static void                 AllocateMultiData(std::tuple<Args*&...> pointers, DeviceMemory& memory,
                                                      const std::array<size_t, sizeof...(Args)>& sizeList,
                                                      size_t alignment = Memory::AlignByteCount);
};

template<class T>
inline constexpr T* DeviceMemoryCPUBacked::DeviceData()
{
    return reinterpret_cast<T*>(d_ptr);
}

template<class T>
inline constexpr const T* DeviceMemoryCPUBacked::DeviceData() const
{
    return reinterpret_cast<T*>(d_ptr);
}

template<class T>
inline constexpr T* DeviceMemoryCPUBacked::HostData()
{
    return reinterpret_cast<T*>(h_ptr);
}

template<class T>
inline constexpr const T* DeviceMemoryCPUBacked::HostData() const
{
    return reinterpret_cast<T*>(h_ptr);
}

template<class T>
inline constexpr DeviceMemory::operator T*()
{
    return reinterpret_cast<T*>(m_ptr);
}

template<class T>
inline constexpr DeviceMemory::operator const T*() const
{
    return reinterpret_cast<T*>(m_ptr);
}

inline constexpr DeviceMemory::operator void*()
{
    return m_ptr;
}

inline constexpr DeviceMemory::operator const void*() const
{
    return m_ptr;
}

inline void DeviceMemory::EnlargeBuffer(DeviceMemory& mem, size_t s)
{
    if(s > mem.Size())
    {
        mem = std::move(DeviceMemory());
        mem = std::move(DeviceMemory(s));
    }
}

namespace DeviceMemDetail
{
    template<size_t I = 0, class... Tp>
    inline typename std::enable_if<I == sizeof...(Tp), size_t>::type
    AcquireTotalSize(std::array<size_t, sizeof...(Tp)>& alignedSizeList,
                     const std::array<size_t, sizeof...(Tp)>& countList,
                     size_t alignment)
    {
        return 0;
    }

    template<std::size_t I = 0, class... Tp>
    inline typename std::enable_if<(I < sizeof...(Tp)), size_t>::type
    AcquireTotalSize(std::array<size_t, sizeof...(Tp)>& alignedSizeList,
                     const std::array<size_t, sizeof...(Tp)>& countList,
                     size_t alignment)
    {
        using CurrentType = typename std::tuple_element_t<I, std::tuple<Tp...>>;

        size_t alignedSize = Memory::AlignSize(sizeof(CurrentType) * countList[I],
                                        alignment);

        alignedSizeList[I] = alignedSize;

        return alignedSize + AcquireTotalSize<I + 1, Tp...>(alignedSizeList, countList, alignment);
    }

    template<std::size_t I = 0, class... Tp>
    inline typename std::enable_if<I == sizeof...(Tp), void>::type
    CalculatePointers(std::tuple<Tp*&...>& t, size_t& offset, Byte* memory,
                      const std::array<size_t, sizeof...(Tp)>& alignedSizeList)
    {}

    template<std::size_t I = 0, class... Tp>
    inline typename std::enable_if<(I < sizeof...(Tp)), void>::type
    CalculatePointers(std::tuple<Tp*&...>& t, size_t& offset, Byte* memory,
                      const std::array<size_t, sizeof...(Tp)>& alignedSizeList)
    {
        using CurrentType = typename std::tuple_element_t<I, std::tuple<Tp...>>;
        // Set Pointer
        size_t size = alignedSizeList[I];
        std::get<I>(t) = (size == 0) ? nullptr : reinterpret_cast<CurrentType*>(memory + offset);
        // Increment Offset
        offset += size;
        // Statically Recurse
        CalculatePointers<I + 1, Tp...>(t, offset, memory, alignedSizeList);
    }
}

template <class... Args>
void DeviceMemory::AllocateMultiData(std::tuple<Args*&...> pointers, DeviceMemory& memory,
                                     const std::array<size_t, sizeof...(Args)>& countList,
                                     size_t alignment)
{
    std::array<size_t, sizeof...(Args)> alignedSizeList;
    // Acquire total size & allocation size of each array
    size_t totalSize = DeviceMemDetail::AcquireTotalSize<0, Args...>(alignedSizeList,
                                                                     countList,
                                                                     alignment);
    // Allocate Memory
    DeviceMemory::EnlargeBuffer(memory, totalSize);
    Byte* ptr = static_cast<Byte*>(memory);
    // Populate pointers
    size_t offset = 0;
    DeviceMemDetail::CalculatePointers(pointers, offset, ptr, alignedSizeList);

    assert(totalSize == offset);
}