#pragma once
/**

Slightly over-engineered meta pool generation class

*/

#include <tuple>
#include <list>

#include "TracerTypeGenerators.h"
#include "MangledNames.h"
#include "GPUMaterialI.h"
#include "GPUPrimitiveI.h"
#include "GPUEndpointI.h"

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

template <class... Args>
using BoundaryWorkBatchGeneratorFunc = GPUWorkBatchI * (*)(const CPUEndpointGroupI&,
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
            GPUWorkBatchI* workBatch = gFunc(mg, pg, t, args...);
            return WorkGPtr(workBatch, dFunc);
        }
};

template <class... Args>
class GPUBoundaryWorkBatchGen
{
    private:
        BoundaryWorkBatchGeneratorFunc<Args...>     gFunc;
        ObjDestroyerFunc<GPUWorkBatchI>             dFunc;

    protected:
    public:
        GPUBoundaryWorkBatchGen(BoundaryWorkBatchGeneratorFunc<Args...> g,
                                ObjDestroyerFunc<GPUWorkBatchI> d)
            : gFunc(g)
            , dFunc(d)
        {}

        WorkGPtr operator()(const CPUEndpointGroupI& eg,
                            const GPUTransformI* const* t,
                            Args... args)
        {
            GPUWorkBatchI* workBatch = gFunc(eg, t, args...);
            return WorkGPtr(workBatch, dFunc);
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

    template <class Base, class WorkBatch, class... Args>
    Base* BoundaryWorkBatchConstruct(const CPUEndpointGroupI& eg,
                                     const GPUTransformI* const* t,
                                     Args... args)
    {
        return new WorkBatch(eg, t, args...);
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
        TracerError             GenerateWorkBatch(GPUWorkBatchI*&,
                                                  const char* workNameOverride,
                                                  const GPUMaterialGroupI&,
                                                  const GPUPrimitiveGroupI&,
                                                  const GPUTransformI* const*,
                                                  Args...);
};

template <class... Args>
class BoundaryWorkPool
{
    private:
        std::map<std::string, GPUBoundaryWorkBatchGen<Args...>>     generators;
        std::list<WorkGPtr>                                         allocatedResources;

        // Recursive Loop Functions to Append new Generators
        template<std::size_t I = 0, class... Tp>
        inline typename std::enable_if<I == sizeof...(Tp), void>::type
        LoopAndAppend(std::tuple<Tp...>& t);

        template<std::size_t I = 0, class... Tp>
        typename std::enable_if<(I < sizeof...(Tp)), void>::type
        LoopAndAppend(std::tuple<Tp...>& t);

    public:
        // Constructors & Destructor
                                BoundaryWorkPool() = default;
                                ~BoundaryWorkPool() = default;
        // Meta Work Add
        template <class... Batches>
        void                    AppendGenerators(TypeList<Batches...> batches);

        void                    DeleteAllWorkInstances();

        TracerError             GenerateWorkBatch(GPUWorkBatchI*&,
                                                  const CPUEndpointGroupI&,
                                                  const GPUTransformI* const* t,
                                                  Args...);
        TracerError             GenerateWorkBatch(GPUWorkBatchI*&,
                                                  const char* workNameOverride,
                                                  const CPUEndpointGroupI&,
                                                  const GPUTransformI* const*,
                                                  Args...);
};

template <class... Args>
template<size_t I, class... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
WorkPool<Args...>::LoopAndAppend(std::tuple<Tp...>&)
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
    else return TracerError(TracerError::UNABLE_TO_GENERATE_WORK,
                            mangledName);
    return TracerError::OK;
}

template <class... Args>
TracerError WorkPool<Args...>::GenerateWorkBatch(GPUWorkBatchI*& work,
                                                 const char* workNameOverride,
                                                 const GPUMaterialGroupI& mg,
                                                 const GPUPrimitiveGroupI& pg,
                                                 const GPUTransformI* const* t,
                                                 Args... args)
{
    auto loc = generators.end();
    if((loc = generators.find(workNameOverride)) != generators.end())
    {
        auto ptr = loc->second(mg, pg, t, args...);
        work = ptr.get();
        allocatedResources.emplace_back(std::move(ptr));
    }
    else return TracerError(TracerError::UNABLE_TO_GENERATE_WORK,
                            workNameOverride);
    return TracerError::OK;
}

// ===================================== //
//          BOUDARY WORK PORTION         //
// ===================================== //

template <class... Args>
template<size_t I, class... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type
BoundaryWorkPool<Args...>::LoopAndAppend(std::tuple<Tp...>&)
{}

template <class... Args>
template<size_t I, class... Tp>
inline typename std::enable_if<(I < sizeof...(Tp)), void>::type
BoundaryWorkPool<Args...>::LoopAndAppend(std::tuple<Tp...>& t)
{
    using namespace TypeGenWrappers;
    using CurrentType = typename std::tuple_element_t<I, TypeList<Tp...>>::type::type;
    // Accelerator Types
    generators.emplace(CurrentType::TypeName(),
                       GPUBoundaryWorkBatchGen<Args...>(BoundaryWorkBatchConstruct<GPUWorkBatchI, CurrentType, Args...>,
                                                        DefaultDestruct<GPUWorkBatchI>));
   LoopAndAppend<I + 1, Tp...>(t);
}

template <class... Args>
template <class... Batches>
void BoundaryWorkPool<Args...>::AppendGenerators(TypeList<Batches...> batches)
{
    LoopAndAppend(batches);
}

template <class... Args>
inline void BoundaryWorkPool<Args...>::DeleteAllWorkInstances()
{
    allocatedResources.clear();
}

template <class... Args>
TracerError BoundaryWorkPool<Args...>::GenerateWorkBatch(GPUWorkBatchI*& work,
                                                 const CPUEndpointGroupI& eg,
                                                 const GPUTransformI* const* t,
                                                 Args... args)
{
    std::string mangledName = MangledNames::BoundaryWorkBatch(eg.Type());

    auto loc = generators.end();
    if((loc = generators.find(mangledName)) != generators.end())
    {
        auto ptr = loc->second(eg, t, args...);
        work = ptr.get();
        allocatedResources.emplace_back(std::move(ptr));
    }
    else return TracerError(TracerError::UNABLE_TO_GENERATE_WORK, mangledName);
    return TracerError::OK;
}

template <class... Args>
TracerError BoundaryWorkPool<Args...>::GenerateWorkBatch(GPUWorkBatchI*& work,
                                                 const char* workNameOverride,
                                                 const CPUEndpointGroupI& eg,
                                                 const GPUTransformI* const* t,
                                                 Args... args)
{
    auto loc = generators.end();
    if((loc = generators.find(workNameOverride)) != generators.end())
    {
        auto ptr = loc->second(eg, t, args...);
        work = ptr.get();
        allocatedResources.emplace_back(std::move(ptr));
    }
    else return TracerError(TracerError::UNABLE_TO_GENERATE_WORK, workNameOverride);
    return TracerError::OK;
}
