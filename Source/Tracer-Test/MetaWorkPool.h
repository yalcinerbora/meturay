#pragma once
/**

Slightly overengineered meta pool generation class

*/

#include <tuple>
#include <list>

#include "TracerLib/TracerLogicPools.h"
#include "TracerLib/TracerTypeGenerators.h"
#include "TracerLib/MangledNames.h"
#include "TracerLib/GPUMaterialI.h"
#include "TracerLib/GPUPrimitiveI.h"

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

class MetaWorkPool
{
    private:
        std::map<std::string, GPUWorkBatchGen>  generators;
        std::list<WorkGPtr>                     allocatedResources;

        // Recursive Loop Functions to Append new Generators
        template<std::size_t I = 0, class... Tp>
        inline typename std::enable_if<I == sizeof...(Tp), void>::type
        LoopAndAppend(std::tuple<Tp...>& t);

        template<std::size_t I = 0, class... Tp>
        typename std::enable_if<(I < sizeof...(Tp)), void>::type
        LoopAndAppend(std::tuple<Tp...>& t);

    public:
        // Constructors & Destructor
                                MetaWorkPool() = default;
                                ~MetaWorkPool() = default;
        // Meta Add
        template <class... Batches>
        void                    AppendGenerators(TypeList<Batches...> batches);

        void                    DeleteAllWorkInstances();
        TracerError             GenerateWorkBatch(GPUWorkBatchI*&,
                                                  const GPUMaterialGroupI&,
                                                  const GPUPrimitiveGroupI&);
};

template<size_t I, class... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
MetaWorkPool::LoopAndAppend(std::tuple<Tp...>& t)
{}

template<size_t I, class... Tp>
inline typename std::enable_if<(I < sizeof...(Tp)), void>::type
MetaWorkPool::LoopAndAppend(std::tuple<Tp...>& t)
{
    using namespace TypeGenWrappers;
    using CurrentType = typename std::tuple_element_t<I, TypeList<Tp...>>::type::type;
   // Accelerator Types
    generators.emplace(CurrentType::TypeName(),
                       GPUWorkBatchGen(WorkBatchConstruct<GPUWorkBatchI, CurrentType>,
                                       DefaultDestruct<GPUWorkBatchI>));
   LoopAndAppend<I + 1, Tp...>(t);
}

template <class... Batches>
void MetaWorkPool::AppendGenerators(TypeList<Batches...> batches)
{
    LoopAndAppend(batches);
}

inline void MetaWorkPool::DeleteAllWorkInstances()
{
    allocatedResources.clear();
}

inline TracerError MetaWorkPool::GenerateWorkBatch(GPUWorkBatchI*& work,
                                                   const GPUMaterialGroupI& mg,
                                                   const GPUPrimitiveGroupI& pg)
{
    std::string mangledName = MangledNames::WorkBatch(mg.Type(),
                                                      pg.Type());

    auto loc = generators.end();
    if((loc = generators.find(mangledName)) != generators.end())
    {
        auto ptr = loc->second(mg, pg);
        allocatedResources.emplace_back(std::move(ptr));
        work = ptr.get();
    }
    else return TracerError::UNABLE_TO_GENERATE_WORK;
    return TracerError::OK;
}