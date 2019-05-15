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
        static constexpr std::false_type check(...);

        typedef decltype(check<C>(0)) type;

    public:
        static constexpr bool value = type::value;
};

template <class C>
using HasTypeName = HasTypeNameT<C, const char*()>;

template <class T, template <class> class... Ps>
constexpr bool satisfies_all_v = std::conjunction<Ps<T>...>::value;

template <class T>
struct IsTracerClass
{
    static constexpr bool HasTypeName_v = HasTypeName<T>::value;
    static constexpr bool is_class_v = std::is_class<T>::value;
    static constexpr bool not_abstract_v = !std::is_abstract<T>::value;

    // TODO: Add more when required
    static_assert(HasTypeName_v, "A Tracer class should have"
                                 "\"static const char* TypeName()\" function");
    static_assert(is_class_v, "A Tracer class be a class. (duh)");
    static_assert(not_abstract_v, "A Tracer class be a class. (duh)");


    static constexpr bool value = HasTypeName_v && 
                                  is_class_v &&
                                  not_abstract_v;
};