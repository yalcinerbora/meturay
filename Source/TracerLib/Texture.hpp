
static constexpr cudaTextureAddressMode DetermineAddressMode(EdgeResolveType e)
{
    switch(e)
    {
        default:
        case EdgeResolveType::WRAP:
            return cudaTextureAddressMode::cudaAddressModeWrap;
        case EdgeResolveType::CLAMP:
            return cudaTextureAddressMode::cudaAddressModeClamp;
        case EdgeResolveType::MIRROR:
            return cudaTextureAddressMode::cudaAddressModeMirror;
    }
}

static constexpr cudaTextureFilterMode DetermineFilterMode(InterpolationType i)
{
    switch(i)
    {
        default:
        case InterpolationType::NEAREST:
            return cudaTextureFilterMode::cudaFilterModePoint;
        case InterpolationType::LINEAR:
            return cudaTextureFilterMode::cudaFilterModeLinear;
    }

}

template<int D, class T>
Texture<D, T>::Texture(int deviceId,
                       InterpolationType interp,
                       EdgeResolveType eResolve,
                       const Vector<D, unsigned int>& dim)
    : DeviceLocalMemoryI(deviceId)
{
    cudaExtent extent = {};
    if constexpr(D == 1)
    {
        extent = make_cudaExtent(dim[0], 0 , 0);
    }
    else if constexpr(D == 2)
    {
        extent = make_cudaExtent(dim[0], dim[1], 0);
    }
    else if constexpr(D == 3)
    {
        extent = make_cudaExtent(dim[0], dim[1], dim[2]);
    }

    cudaChannelFormatDesc d = cudaCreateChannelDesc<T>();
    CUDA_MEMORY_CHECK(cudaMallocMipmappedArray(&data, &d, extent, 1));

    // Allocation Done now generate texture
    cudaResourceDesc rDesc = {};
    cudaTextureDesc tDesc = {};

    bool unormType = (std::is_same_v<T, char> ||
                      std::is_same_v<T, short> ||
                      std::is_same_v<T, int> ||
                      std::is_same_v<T, unsigned char> ||
                      std::is_same_v<T, unsigned short> ||
                      std::is_same_v<T, unsigned int>);

    rDesc.resType = cudaResourceType::cudaResourceTypeMipmappedArray;
    rDesc.res.mipmap.mipmap = data;

    tDesc.addressMode[0] = DetermineAddressMode(eResolve);
    tDesc.addressMode[1] = DetermineAddressMode(eResolve);
    tDesc.addressMode[2] = DetermineAddressMode(eResolve);
    tDesc.filterMode = DetermineFilterMode(interp);
    tDesc.readMode = (unormType) ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
    tDesc.sRGB = 0;
    tDesc.borderColor[0] = 0.0f;
    tDesc.borderColor[1] = 0.0f;
    tDesc.borderColor[2] = 0.0f;
    tDesc.borderColor[3] = 0.0f;
    tDesc.normalizedCoords = 1;
    tDesc.mipmapFilterMode = DetermineFilterMode(interp);

    //
    tDesc.maxAnisotropy = 0;
    tDesc.mipmapLevelBias = 0.0f;
    tDesc.minMipmapLevelClamp = 0.0f;
    tDesc.maxMipmapLevelClamp = 100.0f;

    CUDA_CHECK(cudaCreateTextureObject(&t, &rDesc, &tDesc, nullptr));
}
template<int D, class T>
Texture<D, T>::~Texture()
{}

template<int D, class T>
void Texture<D, T>::Copy(const Byte* sourceData,
                         const Vector<D, unsigned int>& size,
                         const Vector<D, unsigned int>& offset,
                         int mipLevel)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = make_cudaExtent(size[0], size[1], size[2]);

    p.dstArray = levelArray;
    p.dstPos = make_cudaPos(0, 0, 0);

    p.srcPos = make_cudaPos(0, 0, 0);
    p.srcPtr = make_cudaPitchedPtr(const_cast<Byte*>(sourceData),
                                   size[0] * sizeof(T),
                                   size[0], size[1]);

    CUDA_CHECK(cudaMemcpy3D(&p));
}

template<int D, class T>
GPUFence Texture<D, T>::CopyAsync(const Byte* sourceData,
                                  const Vector<D, unsigned int>& size,
                                  const Vector<D, unsigned int>& offset,
                                  int mipLevel,
                                  cudaStream_t stream)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = make_cudaExtent(size[0], size[1], size[2]);

    p.dstArray = levelArray;
    p.dstPos = make_cudaPos(0, 0, 0);

    p.srcPos = make_cudaPos(0, 0, 0);
    p.srcPtr = make_cudaPitchedPtr(const_cast<Byte*>(sourceData),
                                   size[0] * sizeof(T),
                                   size[0], size[1]);

    CUDA_CHECK(cudaMemcpy3DAsync(&p, stream));
    return GPUFence(stream);
}

template<int D, class T>
void Texture<D, T>::MigrateToOtherDevice(int deviceTo, cudaStream_t stream)
{

}

template<int D, class T>
TextureArray<D, T>::TextureArray(int deviceId,
                                 InterpolationType interp,
                                 EdgeResolveType eResolve,
                                 const Vector<D, unsigned int>& dim,
                                 unsigned int count)
    : DeviceLocalMemoryI(deviceId)
{
    cudaExtent extent = {};
    if constexpr(D == 1)
    {
        extent = make_cudaExtent(dim[0], 0 , 0);
    }
    else if constexpr(D == 2)
    {
        extent = make_cudaExtent(dim[0], dim[1], 0);
    }
    else if constexpr(D == 3)
    {
        extent = make_cudaExtent(dim[0], dim[1], dim[2]);
    }

    cudaChannelFormatDesc d = cudaCreateChannelDesc<T>();
    CUDA_MEMORY_CHECK(cudaMallocMipmappedArray(&data, &d, extent, 1));

    // Allocation Done now generate texture
    cudaResourceDesc rDesc = {};
    cudaTextureDesc tDesc = {};

    bool unormType = (std::is_same_v<T, char> ||
                      std::is_same_v<T, short> ||
                      std::is_same_v<T, int> ||
                      std::is_same_v<T, unsigned char> ||
                      std::is_same_v<T, unsigned short> ||
                      std::is_same_v<T, unsigned int>);

    rDesc.resType = cudaResourceType::cudaResourceTypeMipmappedArray;
    rDesc.res.mipmap.mipmap = data;

    tDesc.addressMode[0] = DetermineAddressMode(eResolve);
    tDesc.addressMode[1] = DetermineAddressMode(eResolve);
    tDesc.addressMode[2] = DetermineAddressMode(eResolve);
    tDesc.filterMode = DetermineFilterMode(interp);
    tDesc.readMode = (unormType) ? cudaReadModeNormalizedFloat : cudaReadModeElementType;
    tDesc.sRGB = 0;
    tDesc.borderColor[0] = 0.0f;
    tDesc.borderColor[1] = 0.0f;
    tDesc.borderColor[2] = 0.0f;
    tDesc.borderColor[3] = 0.0f;
    tDesc.normalizedCoords = 1;
    tDesc.mipmapFilterMode = DetermineFilterMode(interp);

    //
    tDesc.maxAnisotropy = 0;
    tDesc.mipmapLevelBias = 0.0f;
    tDesc.minMipmapLevelClamp = 0.0f;
    tDesc.maxMipmapLevelClamp = 100.0f;

    CUDA_CHECK(cudaCreateTextureObject(&t, &rDesc, &tDesc, nullptr));
}

template<int D, class T>
TextureArray<D, T>::~TextureArray()
{}

template<int D, class T>
void TextureArray<D, T>::Copy(const Byte* sourceData,
                              const Vector<D, unsigned int>& size,
                              int layer,
                              const Vector<D, unsigned int>& offset,
                              int mipLevel)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = make_cudaExtent(size[0], size[1], size[2]);

    p.dstArray = levelArray;
    p.dstPos = make_cudaPos(0, 0, 0);

    p.srcPos = make_cudaPos(0, 0, 0);
    p.srcPtr = make_cudaPitchedPtr(const_cast<Byte*>(sourceData),
                                   size[0] * sizeof(T),
                                   size[0], size[1]);

    CUDA_CHECK(cudaMemcpy3D(&p));
}

template<int D, class T>
GPUFence TextureArray<D, T>::CopyAsync(const Byte* sourceData,
                                       const Vector<D, unsigned int>& size,
                                       int layer,
                                       const Vector<D, unsigned int>& offset,
                                       int mipLevel,
                                       cudaStream_t stream)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = make_cudaExtent(size[0], size[1], size[2]);

    p.dstArray = levelArray;
    p.dstPos = make_cudaPos(0, 0, 0);

    p.srcPos = make_cudaPos(0, 0, 0);
    p.srcPtr = make_cudaPitchedPtr(const_cast<Byte*>(sourceData),
                                   size[0] * sizeof(T),
                                   size[0], size[1]);

    CUDA_CHECK(cudaMemcpy3DAsync(&p, stream));
    return GPUFence(stream);
}

template<int D, class T>
void TextureArray<D, T>::MigrateToOtherDevice(int deviceTo, cudaStream_t stream)
{

}

template <class T>
TextureCube<T>::TextureCube(int deviceId,
                            InterpolationType,
                            EdgeResolveType,
                            const Vector2ui& dim)
{

}

template <class T>
TextureCube<T>::~TextureCube()
{

}

template <class T>
void TextureCube<T>::Copy(const Byte* sourceData,
                          const Vector<2, unsigned int>& size,
                          CubeTexSide,
                          const Vector<2, unsigned int>& offset,
                          int mipLevel)
{

}

template <class T>
GPUFence TextureCube<T>::CopyAsync(const Byte* sourceData,
                                   const Vector<2, unsigned int>& size,
                                   CubeTexSide,
                                   const Vector<2, unsigned int>& offset,
                                   int mipLevel,
                                   cudaStream_t stream)
{

    return GPUFence(stream);
}

template <class T>
void TextureCube<T>::MigrateToOtherDevice(int deviceTo, cudaStream_t stream)
{

}