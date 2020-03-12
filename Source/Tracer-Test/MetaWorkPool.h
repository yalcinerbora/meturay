#pragma once
/**

Slightly overengineered meta pool generation class

*/

#include <tuple>

#include "TracerLib/TracerLogicPools.h"
#include "TracerLib/TracerTypeGenerators.h"
#include "MetaTracerWork.cuh"

// Simple wrapper for tuple usage (we need the type of the tuple
template <class T>
struct TypeListElement
{
    using type = T;
};

template <class... Args>
using TypeList = std::tuple<TypeListElement<Args>...>;

// Work Object Generator List
using WorkBatchGeneratorFunc = GPUWorkBatchI* (*)(const GPUMaterialGroupI&,
                                                  const GPUPrimitiveGroupI&);
using WorkGPtr = SharedLibPtr<GPUWorkBatchI>;

class GPUWorkBatchGen
{
    private:
        WorkBatchGeneratorFunc              gFunc;
        ObjDestroyerFunc<GPUWorkBatchI>     dFunc;
    protected:
    public:
        GPUWorkBatchGen(WorkBatchGeneratorFunc g,
                        ObjDestroyerFunc<GPUWorkBatchI> d)
            : gFunc(g)
            , dFunc(d)
        {}

        WorkGPtr operator()(const GPUMaterialGroupI& mg,
                            const GPUPrimitiveGroupI& pg)
        {
            GPUWorkBatchI* accel = gFunc(mg, pg);
            return WorkGPtr(accel, dFunc);
        }
};

namespace TypeGenWrappers
{
    template <class Base, class WorkBatch>
    Base* WorkBatchConstruct(const GPUMaterialGroupI& mg,
                             const GPUPrimitiveGroupI& pg)
    {
        return new WorkBatch(mg, pg);
    }
}

#include <list>

class MetaWorkPool
{
    private:
        std::map<std::string, GPUWorkBatchGen>  generators;
        std::list<WorkGPtr>                     allocatedResources;

        template<std::size_t I = 0, class... Tp>
        inline typename std::enable_if<I == sizeof...(Tp), void>::type
        LoopAndAppend(std::tuple<Tp...>& t)
        {}

        template<std::size_t I = 0, class... Tp>
        inline typename std::enable_if< I < sizeof...(Tp), void>::type
        LoopAndAppend(std::tuple<Tp...>& t)
        {
            using namespace TypeGenWrappers;
            using CurrentType = typename std::tuple_element_t<I, TypeList<Tp...>>::type::type;
           // Accelerator Types
            generators.emplace(CurrentType::TypeName(),
                               GPUWorkBatchGen(WorkBatchConstruct<GPUWorkBatchI, CurrentType>,
                                               DefaultDestruct<GPUWorkBatchI>));
           LoopAndAppend<I + 1, Tp...>(t);
        }

    public:

        // YOLO
        template <class... Batches>
        void AppendGenerators(TypeList<Batches...> batches)
        {
            LoopAndAppend(batches);
        }


};