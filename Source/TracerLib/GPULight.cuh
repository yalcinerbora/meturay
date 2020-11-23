#pragma once

#include "GPULightI.cuh"

template <class Primitive>
class GPULight : public GPULightI
{


    private:

        __device__ void Sample()
};

class GPULightDirectional : public GPULightI
{

};