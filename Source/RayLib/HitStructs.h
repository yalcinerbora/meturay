#pragma once
/**

*/
#include <cstdint>

#include "CudaCheck.h"
#include "Types.h"

// Auxiliary Ids for transforms and primitives
typedef uint32_t TransformId;
typedef uint64_t PrimitiveId;

static constexpr PrimitiveId INVALID_PRIMITIVE_ID = std::numeric_limits<PrimitiveId>::max();

struct HitStructPtr
{
    private:
        Byte*       dPtr;
        uint32_t    combinedSize;

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

        uint32_t CombinedSize() const { return combinedSize; }

        //template<class T>
        //__device__ __host__
        //const T& operator[](int i) const { return *reinterpret_cast<const T&>(dPtr + combinedSize * i); }
};

template <class T, uint32_t BBits, uint32_t IBits>
struct alignas(sizeof(T)) HitKeyT
{
    using Type = T;

    // Constructors & Destructor
                                    HitKeyT() = default;
    __device__ __host__ constexpr   HitKeyT(T v) : value(v) {}

    // Props
    T                               value;

    __device__ __host__             operator const T&() const;
    __device__ __host__             operator T&();

    __device__ __host__
    static constexpr T              CombinedKey(T batch, T id);
    __device__ __host__
    static constexpr T              FetchIdPortion(HitKeyT key);
    __device__ __host__
    static constexpr T              FetchBatchPortion(HitKeyT key);

    static constexpr uint32_t       BatchBits = BBits;
    static constexpr uint32_t       IdBits = IBits;

    static constexpr T              IdMask = (0x1ull << IdBits) - 1;
    static constexpr T              BatchMask = ((0x1ull << BatchBits) - 1) << IdBits;

    static_assert((IdBits + BatchBits) == std::numeric_limits<T>::digits,
                  "Bits representing portions of HitKey should complement each other.");
    static_assert((IdMask | BatchMask) == std::numeric_limits<T>::max() &&
                  (IdMask & BatchMask) == std::numeric_limits<T>::min(),
                  "Masks representing portions of HitKey should complement each other.");

    static constexpr uint16_t       NullBatch = NullBatchId;
    static constexpr T              InvalidKey = CombinedKey(NullBatch, 0);
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
constexpr T HitKeyT<T, BatchBits, IdBits>::CombinedKey(T batch, T id)
{
    return (static_cast<T>(batch) << IdBits) |
           (static_cast<T>(id) & IdMask);
}

template <class T, uint32_t BatchBits, uint32_t IdBits>
__device__ __host__
constexpr T HitKeyT<T, BatchBits, IdBits>::FetchIdPortion(HitKeyT key)
{
    return key & IdMask;
}

template <class T, uint32_t BatchBits, uint32_t IdBits>
__device__ __host__
constexpr T HitKeyT<T, BatchBits, IdBits>::FetchBatchPortion(HitKeyT key)
{
    return ((static_cast<T>(key) & BatchMask) >> IdBits);
}

// Id-Key pair which will be used in sorting for kernel partitioning
using RayId = uint32_t;

using HitKeyType = uint32_t;
using HitKey = HitKeyT<HitKeyType, 8u, 24u>;

static_assert(sizeof(HitKey) == sizeof(HitKeyType), "Type and Key sizes should match.");