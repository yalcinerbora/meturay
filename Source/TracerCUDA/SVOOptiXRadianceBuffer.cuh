#pragma once

#include "DeviceMemory.h"
#include "RayLib/Vector.h"

#include <type_traits>

class SVOOptixRadianceBuffer
{
    public:
    template <class T>
    struct SegmentedField
    {
        friend class SVOOptixRadianceBuffer;
        static_assert(std::is_pointer_v<T>, "Segmented Field type \"T\" should be a pointer");

        private:
        T               dRadiances; // Linear Radiances
        Vector3i        fieldDim;   // Size of each field (x,y, fieldAmount)

        public:
        // Helper Access
        __device__ __host__
        T               operator[](int32_t fieldIndex);
        //__device__ __host__
        //auto            operator[](int32_t fieldIndex) const  -> std::add_const_t<T>;
        __device__ __host__
        Vector2i        FieldDim() const;
        __device__ __host__
        int32_t        FieldCount() const;
    };

    private:
    DeviceMemory        mem;
    size_t              radianceCount;

    public:
    // Constructors & Destructor
                        SVOOptixRadianceBuffer();
                        SVOOptixRadianceBuffer(int32_t radianceCount);
                        ~SVOOptixRadianceBuffer() = default;

    // Segmentation
    SegmentedField<float*>              Segment(const Vector2i& fieldDim);
    const SegmentedField<const float*>  Segment(const Vector2i& fieldDim) const;

};

inline SVOOptixRadianceBuffer::SVOOptixRadianceBuffer()
    : radianceCount(0)
{}

inline SVOOptixRadianceBuffer::SVOOptixRadianceBuffer(int32_t radianceCount)
    : mem(radianceCount * sizeof(float))
    , radianceCount(radianceCount)
{
    assert(radianceCount >= 0);
}

inline
SVOOptixRadianceBuffer::SegmentedField<float*> SVOOptixRadianceBuffer::Segment(const Vector2i& fieldDim)
{
    int32_t pixelPerField = fieldDim.Multiply();
    int32_t fieldCount = static_cast<int32_t>(radianceCount / pixelPerField);

    Vector3i fieldDim3D = Vector3i(fieldDim, fieldCount);
    SegmentedField<float*> result;
    result.dRadiances = static_cast<float*>(mem);
    result.fieldDim = fieldDim3D;
    return result;
}

inline
const SVOOptixRadianceBuffer::SegmentedField<const float*> SVOOptixRadianceBuffer::Segment(const Vector2i& fieldDim) const
{
    int32_t pixelPerField = fieldDim.Multiply();
    int32_t fieldCount = static_cast<int32_t>(radianceCount / pixelPerField);

    Vector3i fieldDim3D = Vector3i(fieldDim, fieldCount);
    SegmentedField<const float*> result;
    result.dRadiances = static_cast<const float*>(mem);
    result.fieldDim = fieldDim3D;
    return result;
}

template <class T>
__device__ __host__ __forceinline__
T SVOOptixRadianceBuffer::SegmentedField<T>::operator[](int32_t fieldIndex)
{
    assert(fieldIndex <= fieldDim[2]);
    int32_t pixelPerField = fieldDim[0] * fieldDim[1];
    return dRadiances + fieldIndex * pixelPerField;
}

template <class T>
__device__ __host__ __forceinline__
Vector2i SVOOptixRadianceBuffer::SegmentedField<T>::FieldDim() const
{
    return Vector2i(fieldDim);
}

template <class T>
__device__ __host__ __forceinline__
int32_t SVOOptixRadianceBuffer::SegmentedField<T>::FieldCount() const
{
    return fieldDim[2];
}