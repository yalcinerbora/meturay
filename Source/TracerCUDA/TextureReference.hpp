
template <class T>
__device__
inline T ConvertTexReturnType(CudaReturn_t<T> cudaData)
{
    if constexpr(std::is_same_v<T, float>)
        return cudaData;
    else if constexpr(std::is_same_v<T, Vector2f>)
        return Vector2f(cudaData.x, cudaData.y);
    else if constexpr(std::is_same_v<T, Vector3f>)
        return Vector3f(cudaData.x, cudaData.y, cudaData.z);
    else if constexpr(std::is_same_v<T, Vector4f>)
        return Vector4f(cudaData.x, cudaData.y, cudaData.z, cudaData.w);
    else static_assert(false, "Up to 4 channel float can be returned from a texture.");
    return T();
}

template <int D, class T>
__device__ ConstantRef<D, T>::ConstantRef(T d)
 : data(d)
{}

template <int D, class T>
__device__
T ConstantRef<D, T>::operator()(const TexFloatType_t<D>&) const
{
    return data;
}

template <int D, class T>
__device__
T ConstantRef<D, T>::operator()(const TexFloatType_t<D>&, float) const
{
    return data;
}

template <int D, class T>
__device__
T ConstantRef<D, T>::operator()(const TexFloatType_t<D>&,
                                const TexFloatType_t<D>&,
                                const TexFloatType_t<D>&) const
{
    return data;
}

template <int D, class T>
__device__ TextureRef<D, T>::TextureRef(cudaTextureObject_t tId)
 : t(tId)
{}

template <int D, class T>
__device__
T TextureRef<D, T>::operator()(const TexFloatType_t<D>& index) const
{
    if constexpr(D == 1)
    {
        return ConvertTexReturnType<T>(tex1D<CudaReturn_t<T>>(t, index));
    }
    else if constexpr(D == 2)
    {
        T val = ConvertTexReturnType<T>(tex2D<CudaReturn_t<T>>(t, index[0], index[1]));
        printf("T(%f, %f) = (%f, %f, %f)\n", 
               index[0], index[1],
               val[0], val[1], val[2]);
        return val;
        //return ConvertTexReturnType<T>(tex2D<CudaReturn_t<T>>(t, index[0], index[1]));
    }
    else if constexpr(D == 3)
    {
        return ConvertTexReturnType<T>(tex3D<CudaReturn_t<T>>(t, index[0], index[1], index[2]));
    }
    return T();
}

template <int D, class T>
__device__
T TextureRef<D, T>::operator()(const TexFloatType_t<D>& index, float mip) const
{
    if constexpr(D == 1)
    {
        return ConvertTexReturnType<T>(tex1DLod<CudaReturn_t<T>>(t, index, mip));
    }
    else if constexpr(D == 2)
    {
        return ConvertTexReturnType<T>(tex2DLod<CudaReturn_t<T>>(t, index[0], index[1], mip));
    }
    else if constexpr(D == 3)
    {
        return ConvertTexReturnType<T>(tex3DLod<CudaReturn_t<T>>(t, index[0], index[1], index[2], mip));
    }
    return T();
}

template <int D, class T>
__device__
T TextureRef<D, T>::operator()(const TexFloatType_t<D>& index,
                               const TexFloatType_t<D>& dx,
                               const TexFloatType_t<D>& dy) const
{
    if constexpr(D == 1)
    {
        return ConvertTexReturnType<T>(tex1DGrad<CudaReturn_t<T>>(t, index, dx, dy));
    }
    else if constexpr(D == 2)
    {
        return ConvertTexReturnType<T>(tex2DGrad<CudaReturn_t<T>>(t, index[0], index[1],
                                                                  float2{dx[0], dx[1]},
                                                                  float2{dy[0], dy[1]}));
    }
    else if constexpr(D == 3)
    {
        return ConvertTexReturnType<T>(tex3DGrad<CudaReturn_t<T>>(t, index[0], index[1], index[2],
                                                                  float4{dx[0], dx[1], dx[2], 0},
                                                                  float4{dy[0], dy[1], dy[2], 0}));
    }
    return T();
}