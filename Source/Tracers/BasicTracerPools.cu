#include "BasicTracerPools.h"

#include "DirectTracer.h"
#include "PathTracer.h"

using namespace TypeGenWrappers;

BasicTracerPool::BasicTracerPool()
{
    tracerGenerators.emplace(DirectTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, DirectTracer>,
                                          DefaultDestruct<GPUTracerI>));
    tracerGenerators.emplace(PathTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, PathTracer>,
                                          DefaultDestruct<GPUTracerI>));
}