#pragma once
/**


*/
#include <cstdint>

#include "CudaCheck.h"
#include "Types.h"

// Auxiliary Ids for transforms and primitives
typedef uint32_t TransformId;
typedef uint64_t PrimitiveId;

// TODO: Implement
struct HitStructPtr
{
    private:
        Byte*       dPtr;
        int         combinedSize;

    public:
                    HitStructPtr() : dPtr(nullptr), combinedSize(0) {}
                    HitStructPtr(void* dPtr, int combinedSize)
                        : dPtr(static_cast<Byte*>(dPtr))
                        , combinedSize(combinedSize)
                    {}

        template<class T>
        __device__ __host__
        T& Ref(int i) { return *reinterpret_cast<T*>(dPtr + combinedSize * i); }
        template<class T>
        __device__ __host__
        const T& Ref(int i) const { return *reinterpret_cast<T*>(dPtr + combinedSize * i); }

        //template<class T>
        //__device__ __host__
        //const T& operator[](int i) const { return *reinterpret_cast<const T&>(dPtr + combinedSize * i); }
};

template <class T, uint32_t BatchBits, uint32_t IdBits>
struct alignas(sizeof(T)) HitKeyT
{
    using Type = T;

    // Constructors & Destructor
                                    HitKeyT() = default;
    __device__ __host__             HitKeyT(T v) : value(v) {}

    // Props
    uint32_t                        value;

    __device__ __host__             operator const T&() const;
    __device__ __host__             operator T&();

    __device__ __host__
    static constexpr T              CombinedKey(uint32_t batch, uint64_t id);
    __device__ __host__
    static constexpr T              FetchIdPortion(HitKeyT key);
    __device__ __host__
    static constexpr uint16_t       FetchBatchPortion(HitKeyT key);

    static constexpr uint32_t       BatchBits = BatchBits;
    static constexpr uint32_t       IdBits = IdBits;

    static constexpr T              IdMask = (0x1ull << IdBits) - 1;
    static constexpr T              BatchMask = ((0x1ull << BatchBits) - 1) << IdBits;

    static_assert((IdBits + BatchBits) == std::numeric_limits<T>::digits,
                  "Bits representing portions of HitKey should complement each other.");
    static_assert((IdMask | BatchMask) == std::numeric_limits<T>::max() &&
                  (IdMask & BatchMask) == std::numeric_limits<T>::min(),
                  "Masks representing portions of HitKey should complement each other.");


    static constexpr uint16_t       NullBatch = NullBatchId;
    static constexpr uint16_t       BoundaryBatch = BoundaryBatchId;
    static constexpr T              InvalidKey = CombinedKey(NullBatch, 0);
    static constexpr T              BoundaryMatKey = CombinedKey(BoundaryBatch, 0);
};

template <class T, uint32_t BatchBits, uint32_t IdBits>
__device__ __host__ HitKeyT<T, BatchBits, IdBits>::operator const T&() const
{
    return  value;
}

template <class T, uint32_t BatchBits, uint32_t IdBits>
__device__ __host__ HitKeyT<T, BatchBits, IdBits>::operator T&()
{
    return value;
}

template <class T, uint32_t BatchBits, uint32_t IdBits>
__device__ __host__
constexpr T HitKeyT<T, BatchBits, IdBits>::CombinedKey(uint32_t batch, uint64_t id)
{
    return (static_cast<T>(batch) << IdBits) | (id & BatchMask);
}

template <class T, uint32_t BatchBits, uint32_t IdBits>
__device__ __host__
constexpr T HitKeyT<T, BatchBits, IdBits>::FetchIdPortion(HitKeyT key)
{
    return key & IdMask;
}

template <class T, uint32_t BatchBits, uint32_t IdBits>
__device__ __host__
constexpr  uint16_t HitKeyT<T, BatchBits, IdBits>::FetchBatchPortion(HitKeyT key)
{
    return static_cast<uint16_t>((static_cast<T>(key) & BatchMask) >> IdBits);
}

// Id-Key pair which will be used in sorting for kernel partitioning
typedef uint32_t RayId;

using HitKeyType = uint32_t;
using HitKey = HitKeyT<HitKeyType, 8, 24>;

static_assert(sizeof(HitKey) == sizeof(HitKeyType), "Type and Key sizes should match.");