#pragma once

#include <memory>

// Load Base Instace
template <class T>
using ObjGeneratorFunc = T* (*)();

template <class T>
using ObjDestroyerFunc = void(*)(T*);

template <class T>
using SharedLibPtr = std::unique_ptr<T, ObjDestroyerFunc<T>>;