#pragma once
/**

Tracer Data Structure Type trait

*/

#include <type_traits>

template<typename, typename T>
struct HasTypeNameT
{
    static_assert(std::integral_constant<T, false>::value,
                  "Second template parameter needs to be of function type.");
};

template<typename C, typename Ret, typename... Args>
struct HasTypeNameT<C, Ret(Args...)>
{
    private:
        template<typename T>
        static constexpr auto check(T*) -> typename
            std::is_same<decltype(T::TypeName(std::declval<Args>()...)), Ret>::type;
                                  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                  // attempt to call it and see if the return type is correct

        template<typename>
        static constexpr std::false_type check(...) { return std::false_type(); }

        typedef decltype(check<C>(0)) type;

    public:
        static constexpr bool value = type::value;
};

template <class C>
using HasTypeName = HasTypeNameT<C, const char*()>;

template<typename C, typename Ret, typename ...Args>
template<typename T>
inline constexpr auto HasTypeNameT<C, Ret(Args...)>::check(T*) -> typename std::is_same<decltype(T::TypeName(std::declval<Args>()...)), Ret>::type
{
    return typename std::is_same<decltype(T::TypeName(std::declval<Args>()...)), Ret>::type();
}

template <class T>
struct IsTracerClass
{
    static constexpr bool has_type_name_v = HasTypeName<T>::value;
    static constexpr bool is_class_v = std::is_class<T>::value;
    static constexpr bool not_abstract_v = !std::is_abstract<T>::value;

    // TODO: Add more when required
    static_assert(has_type_name_v, "A Tracer class should have"
                  "\"static const char* TypeName()\" function");
    static_assert(is_class_v, "A Tracer class must be a class. (duh)");
    static_assert(not_abstract_v, "A Tracer class must not be a abstract class. (duh)");

    static constexpr bool value = has_type_name_v &&
                                  is_class_v &&
                                  not_abstract_v;
};

template <class T>
struct IsMaterialGroupClass
{
    static constexpr bool has_proper_constructor_v = std::is_constructible<T, const CudaGPU&>::value;
    static_assert(has_proper_constructor_v, "A Material Group class should have"
                  "\"MaterialGroup(const CudaGPU&)\" constructor");

    static constexpr bool value = IsTracerClass<T>::value &&
                                  has_proper_constructor_v;
};

template <class T>
struct IsAcceleratorGroupClass
{
    static constexpr bool has_proper_constructor_v = std::is_constructible<T, const GPUPrimitiveGroupI&>::value;
    static_assert(has_proper_constructor_v, "An Accelerator Group class should have"
                  "\"AcceleratorGroup(const GPUPrimitiveGroupI&)\" constructor");

    static constexpr bool value = IsTracerClass<T>::value &&
                                  has_proper_constructor_v;
};

template <class T>
struct IsLightGroupClass
{
    static constexpr bool has_proper_constructor_v = std::is_constructible<T, const GPUPrimitiveGroupI*>::value;
    static_assert(has_proper_constructor_v, "A Light Group class should have"
                  "\"LightGroup(const GPUPrimitiveGroupI*)\" constructor");

    static constexpr bool value = IsTracerClass<T>::value &&
                                  has_proper_constructor_v;
};