#pragma once
/**

Slightly overengineered meta pool generation class

*/

#include <tuple>
#include <list>

#include "TracerLogicPools.h"
#include "TracerTypeGenerators.h"
#include "MangledNames.h"
#include "GPUMaterialI.h"
#include "GPUPrimitiveI.h"

// Simple wrapper for tuple usage (we need the type of the element)
template <class T>
struct TypeListElement
{
    using type = T;
};

template <class... Args>
using TypeList = std::tuple<TypeListElement<Args>...>;

// Work Object Generator List
template <class... Args>
using WorkBatchGeneratorFunc = GPUWorkBatchI* (*)(const GPUMaterialGroupI&,
                                                  const GPUPrimitiveGroupI&,
                                                  const GPUTransformI* const*,
                                                  Args...);
using WorkGPtr = SharedLibPtr<GPUWorkBatchI>;

template <class... Args>
class GPUWorkBatchGen
{
    private:
        WorkBatchGeneratorFunc<Args...>     gFunc;
        ObjDestroyerFunc<GPUWorkBatchI>     dFunc;

    protected:
    public:
        GPUWorkBatchGen(WorkBatchGeneratorFunc<Args...> g,
                        ObjDestroyerFunc<GPUWorkBatchI> d)
            : gFunc(g)
            , dFunc(d)
        {}

        WorkGPtr operator()(const GPUMaterialGroupI& mg,
                            const GPUPrimitiveGroupI& pg,
                            const GPUTransformI* const* t,
                            Args... args)
        {
            GPUWorkBatchI* accel = gFunc(mg, pg, t, args...);
            return WorkGPtr(accel, dFunc);
        }
};

namespace TypeGenWrappers
{
    template <class Base, class WorkBatch, class... Args>
    Base* WorkBatchConstruct(const GPUMaterialGroupI& mg,
                             const GPUPrimitiveGroupI& pg,
                             const GPUTransformI* const* t,
                             Args... args)
    {
        return new WorkBatch(mg, pg, t, args...);
    }
}

template <class... Args>
class WorkPool
{
    private:
        std::map<std::string, GPUWorkBatchGen<Args...>>     generators;
        std::list<WorkGPtr>                                 allocatedResources;

        // Recursive Loop Functions to Append new Generators
        template<std::size_t I = 0, class... Tp>
        inline typename std::enable_if<I == sizeof...(Tp), void>::type
        LoopAndAppend(std::tuple<Tp...>& t);

        template<std::size_t I = 0, class... Tp>
        typename std::enable_if<(I < sizeof...(Tp)), void>::type
        LoopAndAppend(std::tuple<Tp...>& t);

    public:
        // Constructors & Destructor
                                WorkPool() = default;
                                ~WorkPool() = default;
        // Meta Work Add
        template <class... Batches>
        void                    AppendGenerators(TypeList<Batches...> batches);

        void                    DeleteAllWorkInstances();

        TracerError             GenerateWorkBatch(GPUWorkBatchI*&,
                                                  const GPUMaterialGroupI&,
                                                  const GPUPrimitiveGroupI&,
                                                  const GPUTransformI* const* t,
                                                  Args...);
};

template <class... Args>
template<size_t I, class... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
WorkPool<Args...>::LoopAndAppend(std::tuple<Tp...>& t)
{}

template <class... Args>
template<size_t I, class... Tp>
inline typename std::enable_if<(I < sizeof...(Tp)), void>::type
WorkPool<Args...>::LoopAndAppend(std::tuple<Tp...>& t)
{
    using namespace TypeGenWrappers;
    using CurrentType = typename std::tuple_element_t<I, TypeList<Tp...>>::type::type;
    // Accelerator Types
    generators.emplace(CurrentType::TypeName(),
                       GPUWorkBatchGen<Args...>(WorkBatchConstruct<GPUWorkBatchI, CurrentType, Args...>,
                                                DefaultDestruct<GPUWorkBatchI>));
   LoopAndAppend<I + 1, Tp...>(t);
}

template <class... Args>
template <class... Batches>
void WorkPool<Args...>::AppendGenerators(TypeList<Batches...> batches)
{
    LoopAndAppend(batches);
}

template <class... Args>
inline void WorkPool<Args...>::DeleteAllWorkInstances()
{
    allocatedResources.clear();
}

template <class... Args>
TracerError WorkPool<Args...>::GenerateWorkBatch(GPUWorkBatchI*& work,
                                                 const GPUMaterialGroupI& mg,
                                                 const GPUPrimitiveGroupI& pg,
                                                 const GPUTransformI* const* t,
                                                 Args... args)
{
    std::string mangledName = MangledNames::WorkBatch(pg.Type(),
                                                      mg.Type());

    auto loc = generators.end();
    if((loc = generators.find(mangledName)) != generators.end())
    {
        auto ptr = loc->second(mg, pg, t, args...);        
        work = ptr.get();
        allocatedResources.emplace_back(std::move(ptr));
    }
    else return TracerError::UNABLE_TO_GENERATE_WORK;
    return TracerError::OK;
}