#pragma once

#include <cuda.h>
#include <RayLib/Vector.h>

// GPU Texture objec w

// I am not good at SFINAE
// Here we go
template <class T>
struct CudaTexType {};
template <>
struct CudaTexType<float> { using type = float; };
template <>
struct CudaTexType<Vector2f> { using type = float2; };
// Cuda does not support 3 channel texture use 4 channel omit w
// Texture implementation should check this
template <>
struct CudaTexType<Vector3f> { using type = float4; };
template <>
struct CudaTexType<Vector4f> { using type = float4; };
template <class T>
using CudaReturn_t = typename CudaTexType<T>::type;

template <int D>
struct TexFloatType {};
template <>
struct TexFloatType<1> { using type = float; };
template <>
struct TexFloatType<2> { using type = Vector2f; };
template <>
struct TexFloatType<3> { using type = Vector3f; };
template <int D>
using TexFloatType_t = typename TexFloatType<D>::type;

// Float1/2/3 type to Vector1/2/3 type wrappers
template <class T>
__device__
T ConvertTexReturnType(CudaReturn_t<T>);

template <int D, class T>
class TextureRef
{
    public:
        virtual     ~TextureRef() = default;

        __device__
        virtual T   operator()(const TexFloatType_t<D>&) const = 0;
        // Mip Level
        __device__
        virtual T   operator()(const TexFloatType_t<D>&, float mip) const = 0;
        // Gradient
        __device__
        virtual T   operator()(const TexFloatType_t<D>&,
                               const TexFloatType_t<D>& dx,
                               const TexFloatType_t<D>& dy) const = 0;
};

template <int D, class T>
class ConstantRef : public TextureRef<D, T>
{
    private:
        T               data;

    public:
        // Constructors & Destructor
        __device__      ConstantRef(T);
                        ~ConstantRef() = default;

        __device__
        T               operator()(const TexFloatType_t<D>&) const  override;
        __device__
        T               operator()(const TexFloatType_t<D>&, float mip) const  override;
        __device__
        T               operator()(const TexFloatType_t<D>&,
                                   const TexFloatType_t<D>& dx,
                                   const TexFloatType_t<D>& dy) const override;

};

template <int D, class T>
class TexRef : public TextureRef<D, T>
{
    using CudaType = typename CudaReturn_t<T>;

    private:
        cudaTextureObject_t t;

    public:
        // Constructors & Destructor
        __device__      TexRef(cudaTextureObject_t);
                        ~TexRef() = default;

        __device__
        T               operator()(const TexFloatType_t<D>&) const  override;
        // Mip Level
        __device__
        T               operator()(const TexFloatType_t<D>&, float mip) const  override;
        // Gradient
        __device__
        T               operator()(const TexFloatType_t<D>&,
                                   const TexFloatType_t<D>& dx,
                                   const TexFloatType_t<D>& dy) const override;

};

template <int D, class T>
class TexArrayRef : public TextureRef<D, T>
{
    private:
        cudaTextureObject_t     t;
        int                     arrayIndex;

    public:
        __device__      TexArrayRef(cudaTextureObject_t, int);
                        ~TexArrayRef() = default;

        __device__
        T               operator()(const TexFloatType_t<D>&) const  override;
        // Mip Level
        __device__
        T               operator()(const TexFloatType_t<D>&, float mip) const  override;
        // Gradient
        __device__
        T               operator()(const TexFloatType_t<D>&,
                                   const TexFloatType_t<D>& dx,
                                   const TexFloatType_t<D>& dy) const override;

};

template <class T>
class TexCubeRef : public TextureRef<3, T>
{
    private:
        cudaTextureObject_t     t;
        int                     arrayIndex;

    public:
        __device__      TexCubeRef(cudaTextureObject_t);
                        ~TexCubeRef() = default;

        __device__
        T               operator()(const TexFloatType_t<3>&) const  override;
        // Mip Level
        __device__
        T               operator()(const TexFloatType_t<3>&, float mip) const  override;
        // Gradient
        __device__
        T               operator()(const TexFloatType_t<3>&,
                                   const TexFloatType_t<3>& dx,
                                   const TexFloatType_t<3>& dy) const override;
};

#include "TextureReference.hpp"

extern template class ConstantRef<1, float>;
extern template class ConstantRef<1, Vector2>;
extern template class ConstantRef<1, Vector3>;
extern template class ConstantRef<1, Vector4>;

extern template class ConstantRef<2, float>;
extern template class ConstantRef<2, Vector2>;
extern template class ConstantRef<2, Vector3>;
extern template class ConstantRef<2, Vector4>;

extern template class ConstantRef<3, float>;
extern template class ConstantRef<3, Vector2>;
extern template class ConstantRef<3, Vector3>;
extern template class ConstantRef<3, Vector4>;

extern template class TexRef<1, float>;
extern template class TexRef<1, Vector2>;
extern template class TexRef<1, Vector3>;
extern template class TexRef<1, Vector4>;

extern template class TexRef<2, float>;
extern template class TexRef<2, Vector2>;
extern template class TexRef<2, Vector3>;
extern template class TexRef<2, Vector4>;

extern template class TexRef<3, float>;
extern template class TexRef<3, Vector2>;
extern template class TexRef<3, Vector3>;
extern template class TexRef<3, Vector4>;