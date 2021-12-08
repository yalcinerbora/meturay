#pragma once

#include <memory>

// Load Base Instance
template <class T>
using ObjGeneratorFunc = T* (*)();

template <class T, class... Args>
using ObjGeneratorFuncArgs = T* (*)(Args...);

template <class T>
using ObjDestroyerFunc = void(*)(T*);

template <class T>
using SharedLibPtr = std::unique_ptr<T, ObjDestroyerFunc<T>>;

//=========================//
// Shared Ptr Construction //
//=========================//
// Tracer
template <class Interface>
class GeneratorNoArg
{
    private:
    ObjGeneratorFunc<Interface> gFunc;
    ObjDestroyerFunc<Interface> dFunc;

    public:
        // Constructor & Destructor
    GeneratorNoArg(ObjGeneratorFunc<Interface> g,
                   ObjDestroyerFunc<Interface> d)
        : gFunc(g)
        , dFunc(d)
    {}

    SharedLibPtr<Interface> operator()()
    {
        Interface* prim = gFunc();
        return SharedLibPtr<Interface>(prim, dFunc);
    }
};

namespace TypeGenWrappers
{
    //==============//
    // New Wrappers //
    //==============//
    template <class Base, class T>
    Base* DefaultConstruct()
    {
        return new T();
    }

    template <class T>
    void DefaultDestruct(T* t)
    {
        if(t) delete t;
    }

    template <class T>
    void EmptyDestruct(T*) {}
}