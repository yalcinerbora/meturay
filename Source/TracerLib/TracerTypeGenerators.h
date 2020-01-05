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
#include "RayLib/ObjectFuncDefinitions.h"
#include "RayLib/SceneStructs.h"
#include "RayLib/TracerStructs.h"

class TracerBaseLogicI;
class GPUEventEstimatorI;

class GPUBaseAcceleratorI;

class GPUAcceleratorGroupI;
class GPUPrimitiveGroupI;
class GPUMaterialGroupI;

class GPUAcceleratorBatchI;
class GPUMaterialBatchI;

class CudaGPU;


// Fundamental Pointers whichalso support creation-deletion
// across dll boundaries
using GPUTracerPtr = SharedLibPtr<TracerBaseLogicI>;
using GPUBaseAccelPtr = SharedLibPtr<GPUBaseAcceleratorI>;

using GPUAccelGPtr = SharedLibPtr<GPUAcceleratorGroupI>;
using GPUPrimGPtr = SharedLibPtr<GPUPrimitiveGroupI>;
using GPUMatGPtr = SharedLibPtr<GPUMaterialGroupI>;

using GPUAccelBPtr = SharedLibPtr<GPUAcceleratorBatchI>;
using GPUMatBPtr = SharedLibPtr<GPUMaterialBatchI>;

using GPUEstimatorPtr = SharedLibPtr<GPUEventEstimatorI>;

// Statically Inerfaced Generators
template<class TracerLogic>
using TracerLogicGeneratorFunc = TracerLogic* (*)(GPUBaseAcceleratorI& baseAccelerator,
                                                  AcceleratorGroupList&& ag,
                                                  AcceleratorBatchMappings&& ab,
                                                  MaterialGroupList&& mg,
                                                  MaterialBatchMappings&& mb,
                                                  //
                                                  const TracerParameters& params,
                                                  uint32_t hitStructSize,
                                                  const Vector2i maxMats,
                                                  const Vector2i maxAccels,
                                                  const HitKey baseBoundMatKey);

template<class Accel>
using AccelGroupGeneratorFunc = Accel* (*)(const GPUPrimitiveGroupI&,
                                           const TransformStruct*);

template<class AccelBatch>
using AccelBatchGeneratorFunc = AccelBatch* (*)(const GPUAcceleratorGroupI&,
                                                const GPUPrimitiveGroupI&);

template<class MaterialGroup>
using MaterialGroupGeneratorFunc = MaterialGroup* (*)(const CudaGPU& gpuId,
                                                      const GPUEventEstimatorI& e);

template<class MaterialBatch>
using MaterialBatchGeneratorFunc = MaterialBatch* (*)(const GPUMaterialGroupI&,
                                                      const GPUPrimitiveGroupI&);

using GPUBaseAccelGen = GeneratorNoArg<GPUBaseAcceleratorI>;
using GPUPrimGroupGen = GeneratorNoArg<GPUPrimitiveGroupI>;
using GPUEstimatorGen = GeneratorNoArg<GPUEventEstimatorI>;


class GPUTracerGen
{
    private:
        TracerLogicGeneratorFunc<TracerBaseLogicI>  gFunc;
        ObjDestroyerFunc<TracerBaseLogicI>          dFunc;

    public:
        // Constructor & Destructor
        GPUTracerGen(TracerLogicGeneratorFunc<TracerBaseLogicI> g,
                     ObjDestroyerFunc<TracerBaseLogicI> d)
            : gFunc(g)
            , dFunc(d)
        {}

        GPUTracerPtr operator()(GPUBaseAcceleratorI& ba,
                                AcceleratorGroupList&& ag,
                                AcceleratorBatchMappings&& ab,
                                MaterialGroupList&& mg,
                                MaterialBatchMappings&& mb,
                                //
                                const TracerParameters& op,
                                uint32_t hitStructSize,
                                const Vector2i maxMats,
                                const Vector2i maxAccels,
                                const HitKey baseBoundMatKey)
        {
            TracerBaseLogicI* logic = gFunc(ba,
                                            std::move(ag), std::move(ab),
                                            std::move(mg), std::move(mb),
                                            op, hitStructSize,
                                            maxMats, maxAccels,
                                            baseBoundMatKey);
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

        GPUMatGPtr operator()(const CudaGPU& gpu,
                              const GPUEventEstimatorI& e)
        {
            GPUMaterialGroupI* mat = gFunc(gpu, e);
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

        GPUAccelGPtr operator()(const GPUPrimitiveGroupI& pg,
                                const TransformStruct* ts)
        {
            GPUAcceleratorGroupI* accel = gFunc(pg, ts);
            return GPUAccelGPtr(accel, dFunc);
        }
};
// Batch
class GPUAccelBatchGen
{
    private:
        AccelBatchGeneratorFunc<GPUAcceleratorBatchI>   gFunc;
        ObjDestroyerFunc<GPUAcceleratorBatchI>          dFunc;

    public:
        // Constructor & Destructor
        GPUAccelBatchGen(AccelBatchGeneratorFunc<GPUAcceleratorBatchI> g,
                         ObjDestroyerFunc<GPUAcceleratorBatchI> d)
            : gFunc(g)
            , dFunc(d)
        {}

        GPUAccelBPtr operator()(const GPUAcceleratorGroupI& ag,
                                const GPUPrimitiveGroupI& pg)
        {
            GPUAcceleratorBatchI* accel = gFunc(ag, pg);
            return GPUAccelBPtr(accel, dFunc);
        }
};

class GPUMatBatchGen
{
    private:
        MaterialBatchGeneratorFunc<GPUMaterialBatchI>   gFunc;
        ObjDestroyerFunc<GPUMaterialBatchI>             dFunc;

    public:
        // Constructor & Destructor
        GPUMatBatchGen(MaterialBatchGeneratorFunc<GPUMaterialBatchI> g,
                       ObjDestroyerFunc<GPUMaterialBatchI> d)
            : gFunc(g)
            , dFunc(d)
        {}

        GPUMatBPtr operator()(const GPUMaterialGroupI& mg,
                              const GPUPrimitiveGroupI& pg)
        {
            GPUMaterialBatchI* mat = gFunc(mg, pg);
            return GPUMatBPtr(mat, dFunc);
        }
};

namespace TypeGenWrappers
{
    template <class Base, class TracerLogic>
    Base* TracerLogicConstruct(GPUBaseAcceleratorI& ba,
                               AcceleratorGroupList&& ag,
                               AcceleratorBatchMappings&& ab,
                               MaterialGroupList&& mg,
                               MaterialBatchMappings&& mb,

                               const TracerParameters& op,
                               uint32_t hitStrctSize,
                               const Vector2i maxMats,
                               const Vector2i maxAccels,
                               const HitKey baseBoundMatKey)
    {
        return new TracerLogic(ba,
                               std::move(ag), std::move(ab),
                               std::move(mg), std::move(mb),
                               op, hitStrctSize,
                               maxMats, maxAccels,
                               baseBoundMatKey);
    }

    template <class Base, class AccelGroup>
    Base* AccelGroupConstruct(const GPUPrimitiveGroupI& p,
                              const TransformStruct* t)
    {
        return new AccelGroup(p, t);
    }

    template <class Base, class AccelBatch>
    Base* AccelBatchConstruct(const GPUAcceleratorGroupI& a,
                              const GPUPrimitiveGroupI& p)
    {
        return new AccelBatch(a, p);
    }

    template <class Base, class MatBatch>
    Base* MaterialGroupConstruct(const CudaGPU& gpu,
                                 const GPUEventEstimatorI& e)
    {
        return new MatBatch(gpu, e);
    }

    template <class Base, class MatBatch>
    Base* MaterialBatchConstruct(const GPUMaterialGroupI& m,
                                 const GPUPrimitiveGroupI& p)
    {
        return new MatBatch(m, p);
    }
}