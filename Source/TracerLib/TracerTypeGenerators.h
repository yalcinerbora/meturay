#pragma once
/**

Default type generation functions

Most of the time each DLL will came with its own
Class construction functions (which means calling constructor would suffice)

These functions define how generators should be defined. If a type will be generated
accross DLL boundaries it should declare

It also hints how a constructor should be defined. Accelerator should take primitive
as an input. (since those types are storngly tied)


*/
#include "RayLib/SceneStructs.h"
#include "RayLib/TracerStructs.h"

class GPUTracerI;
class GPUSceneI;
class CudaGPU;
class CudaSystem;

// Statically Inerfaced Generators
template<class TracerLogic>
using TracerGeneratorFunc = GPUTracerI* (*)(const CudaSystem&,
                                            const GPUSceneI&,
                                            const TracerParameters&);

template<class Accel>
using AccelGroupGeneratorFunc = Accel* (*)(const GPUPrimitiveGroupI&);

template<class MaterialGroup>
using MaterialGroupGeneratorFunc = MaterialGroup* (*)(const CudaGPU& gpuId);

using GPUBaseAccelGen = GeneratorNoArg<GPUBaseAcceleratorI>;
using GPUPrimGroupGen = GeneratorNoArg<GPUPrimitiveGroupI>;

using CPUTransformGen = GeneratorNoArg<CPUTransformGroupI>;
using CPUMediumGen = GeneratorNoArg<CPUMediumGroupI>;

class GPUTracerGen
{
    private:
        TracerGeneratorFunc<GPUTracerI>   gFunc;
        ObjDestroyerFunc<GPUTracerI>      dFunc;

    public:
        // Constructor & Destructor
        GPUTracerGen(TracerGeneratorFunc<GPUTracerI> g,
                     ObjDestroyerFunc<GPUTracerI> d)
            : gFunc(g)
            , dFunc(d)
        {}

        GPUTracerPtr operator()(const CudaSystem& s,
                                const GPUSceneI& scene,
                                const TracerParameters& params)
        {
            GPUTracerI* logic = gFunc(s, scene, params);
            return GPUTracerPtr(logic, dFunc);
        }
};

class GPUMatGroupGen
{
    private:
        MaterialGroupGeneratorFunc<GPUMaterialGroupI>   gFunc;
        ObjDestroyerFunc<GPUMaterialGroupI>             dFunc;

    public:
        // Constructor & Destructor
        GPUMatGroupGen(MaterialGroupGeneratorFunc<GPUMaterialGroupI> g,
                       ObjDestroyerFunc<GPUMaterialGroupI> d)
            : gFunc(g)
            , dFunc(d)
        {}

        GPUMatGPtr operator()(const CudaGPU& gpu)
        {
            GPUMaterialGroupI* mat = gFunc(gpu);
            return GPUMatGPtr(mat, dFunc);
        }
};

class GPUAccelGroupGen
{
    private:
        AccelGroupGeneratorFunc<GPUAcceleratorGroupI>   gFunc;
        ObjDestroyerFunc<GPUAcceleratorGroupI>          dFunc;

    public:
        // Constructor & Destructor
        GPUAccelGroupGen(AccelGroupGeneratorFunc<GPUAcceleratorGroupI> g,
                         ObjDestroyerFunc<GPUAcceleratorGroupI> d)
            : gFunc(g)
            , dFunc(d)
        {}

        GPUAccelGPtr operator()(const GPUPrimitiveGroupI& pg)
        {
            GPUAcceleratorGroupI* accel = gFunc(pg);
            return GPUAccelGPtr(accel, dFunc);
        }
};

namespace TypeGenWrappers
{
    template <class Base, class TracerLogic>
    Base* TracerLogicConstruct(const CudaSystem& s,
                               const GPUSceneI& scene,
                               const TracerParameters& params)
    {
        return new TracerLogic(s, scene, params);
    }

    template <class Base, class AccelGroup>
    Base* AccelGroupConstruct(const GPUPrimitiveGroupI& p)
    {
        return new AccelGroup(p);
    }

    template <class Base, class MatBatch>
    Base* MaterialGroupConstruct(const CudaGPU& gpu)
    {
        return new MatBatch(gpu);
    }
}