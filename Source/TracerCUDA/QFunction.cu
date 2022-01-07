#include "QFunction.cuh"
#include "CudaSystem.h"
#include "CudaSystem.hpp"

void QFunctionCPU::RecalculateDistributions(const CudaSystem&)
{

}

TracerError QFunctionCPU::Initialize(const CudaSystem&)
{
    return TracerError::OK;
}