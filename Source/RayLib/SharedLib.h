#pragma once

/**

Functionality to Load DLLs or SOs

*/

#include <string>
#include <memory>

#include "ObjectFuncDefinitions.h"
#include "DLLError.h"

struct SharedLibArgs
{
    std::string         mangledConstructorName = "\0";
    std::string         mangledDestructorName = "\0";

    bool                operator<(const SharedLibArgs& s) const;
};

inline bool SharedLibArgs::operator<(const SharedLibArgs& s) const
{
    std::less<std::string> less;
    return less(mangledConstructorName, s.mangledConstructorName);
}

class SharedLib
{
    private:
        static constexpr const char* WinDLLExt      = ".dll";
        static constexpr const char* LinuxDLLExt    = ".so";

        // Props
        void*               libHandle;

        // Internal
        void*               GetProcAdressInternal(const std::string& fName);

    protected:
    public:
        // Constructors & Destructor
                            SharedLib(const std::string& libName);
                            SharedLib(const SharedLib&) = delete;
                            SharedLib(SharedLib&&) noexcept;
        SharedLib&          operator=(const SharedLib&) = delete;
        SharedLib&          operator=(SharedLib&&) noexcept;
                            ~SharedLib();

        template <class T>
        DLLError            GenerateObject(SharedLibPtr<T>&,
                                           const SharedLibArgs& mangledNames);
        template <class T, class... Args>
        DLLError            GenerateObjectWithArgs(SharedLibPtr<T>&,
                                                   const SharedLibArgs& mangledNames,
                                                   Args&&...);
};

using PoolKey = std::pair<SharedLib*, SharedLibArgs>;

template <class T>
DLLError SharedLib::GenerateObject(SharedLibPtr<T>& ptr,
                                   const SharedLibArgs& args)
{
    ObjGeneratorFunc<T> genFunc = reinterpret_cast<ObjGeneratorFunc<T>>(GetProcAdressInternal(args.mangledConstructorName));
    ObjDestroyerFunc<T> destFunc = reinterpret_cast<ObjDestroyerFunc<T>>(GetProcAdressInternal(args.mangledDestructorName));
    if(!genFunc) return DLLError::MANGLED_NAME_NOT_FOUND;
    if(!destFunc) return DLLError::MANGLED_NAME_NOT_FOUND;
    ptr = SharedLibPtr<T>(genFunc(), destFunc);
    return DLLError::OK;
}

template <class T, class... Args>
DLLError SharedLib::GenerateObjectWithArgs(SharedLibPtr<T>& ptr,
                                           const SharedLibArgs& mangledNames,
                                           Args&&... args)
{
    ObjGeneratorFuncArgs<T, Args&&...> genFunc = reinterpret_cast<ObjGeneratorFuncArgs<T, Args&&...>>(GetProcAdressInternal(mangledNames.mangledConstructorName));
    if(!genFunc) return DLLError::MANGLED_NAME_NOT_FOUND;
    ObjDestroyerFunc<T> destFunc = reinterpret_cast<ObjDestroyerFunc<T>>(GetProcAdressInternal(mangledNames.mangledDestructorName));
    if(!destFunc) return DLLError::MANGLED_NAME_NOT_FOUND;
    ptr = SharedLibPtr<T>(genFunc(std::forward<Args&&>(args)...), destFunc);
    return DLLError::OK;
}