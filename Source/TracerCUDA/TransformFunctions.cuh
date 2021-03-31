#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "FunctionEnablers.cuh"

template <class T>
class DeviceDivideFunctor
{
    private:
        const T&        gValue;

    protected:
    public:
                        DeviceDivideFunctor(const T& dValue);
                        ~DeviceDivideFunctor() = default;

        __device__ 
        T               operator()(const T& in);
        
};

template <class T>
class DeviceMultiplyFunctor
{
    private:
        const T&        gValue;

    protected:
    public:
                        DeviceMultiplyFunctor(const T& dValue);
                        ~DeviceMultiplyFunctor() = default;

        __device__ 
        T               operator()(const T& in);
        
};

template <class T>
class HostDivideFunctor
{
    private:
        T               value;

    protected:
    public:
                        HostDivideFunctor(const T& hValue);
                        ~HostDivideFunctor() = default;

        __device__ 
        T               operator()(const T& in);
        
};

template <class T>
class HostMultiplyFunctor
{
    private:
        T               value;

    protected:
    public:
                        HostMultiplyFunctor(const T& hValue);
                        ~HostMultiplyFunctor() = default;

        __device__ 
        T               operator()(const T& in);
        
};

template <class T>
inline DeviceDivideFunctor<T>::DeviceDivideFunctor(const T& dValue)
    : gValue(dValue)
{}

template <class T>
__device__
inline T DeviceDivideFunctor<T>::operator()(const T& in)
{
    return in / gValue;
}

template <class T>
inline DeviceMultiplyFunctor<T>::DeviceMultiplyFunctor(const T& dValue)
    : gValue(dValue)
{}

template <class T>
__device__
inline T DeviceMultiplyFunctor<T>::operator()(const T& in)
{
    return in * gValue;
}

template <class T>
inline HostDivideFunctor<T>::HostDivideFunctor(const T& hValue)
    : value(hValue)
{}

template <class T>
__device__
inline T HostDivideFunctor<T>::operator()(const T& in)
{
    return in / value;
}

template <class T>
inline HostMultiplyFunctor<T>::HostMultiplyFunctor(const T& hValue)
    : value(hValue)
{}

template <class T>
__device__
inline T HostMultiplyFunctor<T>::operator()(const T& in)
{
    return in * value;
}