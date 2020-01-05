#pragma once

#include "EstimatorFunctions.cuh"

struct EmptyEstimatorData {};

struct BasicEstimatorData {};

__device__
inline bool EstimateEventEmpty(HitKey&,
                               Vector3&,
                               float&,
                               //
                               const Vector3&,
                               const Vector3&,
                               RandomGPU&,
                               //
                               const EmptyEstimatorData&)
{
    return false;
}

__device__
inline bool EstimateEventBasic(HitKey&,
                               Vector3&,
                               float&,
                               //
                               const Vector3&,
                               const Vector3&,
                               RandomGPU&,
                               //
                               const BasicEstimatorData&)
{
    return false;
}