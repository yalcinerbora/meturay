#include "TestPools.h"

#include "DirectTracer.h"
#include "PathTracer.h"

using namespace TypeGenWrappers;

TestTracerPool::TestTracerPool()
{
    tracerGenerators.emplace(DirectTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, DirectTracer>,
                                          DefaultDestruct<GPUTracerI>));
    tracerGenerators.emplace(PathTracer::TypeName(),
                             GPUTracerGen(TracerLogicConstruct<GPUTracerI, PathTracer>,
                                          DefaultDestruct<GPUTracerI>));
}