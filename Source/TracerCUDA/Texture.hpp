template <int D>
size_t TotalPixelCount(const TexDimType_t<D>& dim)
{
    if constexpr(D == 1)
        return dim;
    else if constexpr(D == 2)
        return static_cast<size_t>(dim[0]) * dim[1];
    else if constexpr(D == 3)
        return static_cast<size_t>(dim[0]) * dim[1] * dim[2];
    else return 0;
}

template <int D>
cudaExtent MakeCudaCopySize(const TexDimType_t<D>& dim)
{
    if constexpr(D == 1)
    {
        return make_cudaExtent(dim, 1, 1);
    }
    else if constexpr(D == 2)
    {
        return make_cudaExtent(dim[0], dim[1], 1);
    }
    else if constexpr(D == 3)
    {
        return make_cudaExtent(dim[0], dim[1], dim[2]);
    }
}

template <int D>
cudaExtent MakeCudaExtent(const TexDimType_t<D>& dim)
{
    if constexpr(D == 1)
    {
        return make_cudaExtent(dim, 0, 0);
    }
    else if constexpr(D == 2)
    {
        return make_cudaExtent(dim[0], dim[1], 0);
    }
    else if constexpr(D == 3)
    {
        return make_cudaExtent(dim[0], dim[1], dim[2]);
    }
}

template <int D>
cudaExtent MakeCudaExtent(const TexDimType_t<D>& dim,
                          unsigned int layer)
{
    if constexpr(D == 1)
    {
        return make_cudaExtent(dim, layer, 0);
    }
    else if constexpr(D == 2)
    {
        return make_cudaExtent(dim[0], dim[1], layer);
    }
}

template <int D>
cudaPos MakeCudaOffset(const TexDimType_t<D>& offset)
{
    if constexpr(D == 1)
    {
        return make_cudaPos(offset, 0, 0);
    }
    else if constexpr(D == 2)
    {
        return make_cudaPos(offset[0], offset[1], 0);
    }
    else if constexpr(D == 3)
    {
        return make_cudaPos(offset[0], offset[1], offset[2]);
    }
}

template <int D>
cudaPos MakeCudaOffset(const TexDimType_t<D>& offset,
                       unsigned int layer)
{
    if constexpr(D == 1)
    {
        return make_cudaPos(offset, layer, 0);
    }
    else if constexpr(D == 2)
    {
        return make_cudaPos(offset[0], offset[1], layer);
    }
}

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

template<int D>
TextureI<D>::TextureI(TextureI&& other)
    : DeviceLocalMemoryI(other.currentDevice)
    , channelCount(other.channelCount)
    , texture(other.texture)
    , dimensions(other.dimensions)
    , mipCount(mipCount)
{
    other.texture = 0;
    other.dimensions = TexDimType<D>::ZERO;
    other.channelCount = 0;
}

template<int D>
TextureI<D>& TextureI<D>::operator=(TextureI&& other)
{
    assert(this != &other);
    texture = other.texture;
    dimensions = other.dimensions;
    channelCount = other.channelCount;
    mipCount = other.mipCount;

    other.channelCount = 0;
    other.texture = 0;
    other.dimensions = TexDimType<D>::ZERO;
}

template<int D>
TextureArrayI<D>::TextureArrayI(TextureArrayI&& other)
    : DeviceLocalMemoryI(other.currentDevice)
    , texture(other.texture)
    , channelCount(other.channelCount)
    , dimensions(other.dimensions)
    , length(other.length)
{
    other.texture = 0;
    other.dimensions = TexDimType<D>::ZERO;
    other.length = 0;
    other.channelCount = 0;
}

template<int D>
TextureArrayI<D>& TextureArrayI<D>::operator=(TextureArrayI&& other)
{
    assert(this != &other);
    texture = other.texture;
    dimensions = other.dimensions;
    length = other.length;
    channelCount = other.channelCount;

    other.texture = 0;
    other.dimensions = TexDimType<D>::ZERO;
    other.length = 0;
    other.channelCount = 0;
}

template<int D, class T>
Texture<D, T>::Texture(const CudaGPU* device,
                       InterpolationType interp,
                       EdgeResolveType eResolve,
                       bool normalizeIntegers,
                       bool normalizeCoordinates,
                       bool convertSRGB,
                       const TexDimType_t<D>& dim,
                       uint32_t mipCount)
    : TextureI<D>(dim, TextureChannelCount<T>::value, device, mipCount)
    , interpType(interp)
    , edgeResolveType(eResolve)
    , normalizeIntegers(normalizeIntegers)
    , normalizeCoordinates(normalizeCoordinates)
    , convertSRGB(convertSRGB)
{
    cudaExtent extent = MakeCudaExtent<D>(this->dimensions);
    cudaChannelFormatDesc d = cudaCreateChannelDesc<ChannelDescType_t<T>>();
    CUDA_CHECK(cudaSetDevice(device->DeviceId()));
    CUDA_MEMORY_CHECK(cudaMallocMipmappedArray(&data, &d, extent, mipCount));

    // Allocation Done now generate texture
    cudaResourceDesc rDesc = {};
    cudaTextureDesc tDesc = {};

    bool unormType = normalizeIntegers;

    rDesc.resType = cudaResourceType::cudaResourceTypeMipmappedArray;
    rDesc.res.mipmap.mipmap = data;

    tDesc.addressMode[0] = DetermineAddressMode(eResolve);
    tDesc.addressMode[1] = DetermineAddressMode(eResolve);
    tDesc.addressMode[2] = DetermineAddressMode(eResolve);
    tDesc.filterMode = DetermineFilterMode(interp);
    tDesc.readMode = (unormType) ? cudaReadModeNormalizedFloat : cudaReadModeElementType;

    tDesc.sRGB = convertSRGB;
    tDesc.borderColor[0] = 0.0f;
    tDesc.borderColor[1] = 0.0f;
    tDesc.borderColor[2] = 0.0f;
    tDesc.borderColor[3] = 0.0f;
    tDesc.normalizedCoords = normalizeCoordinates;
    tDesc.mipmapFilterMode = DetermineFilterMode(interp);

    tDesc.maxAnisotropy = 4;
    tDesc.mipmapLevelBias = 0.0f;
    tDesc.minMipmapLevelClamp = -100.0f;
    tDesc.maxMipmapLevelClamp = 100.0f;

    CUDA_CHECK(cudaCreateTextureObject(&this->texture, &rDesc, &tDesc, nullptr));
}

template<int D, class T>
Texture<D, T>::Texture(Texture&& other)
    : TextureI<D>(std::move(other))
    , data(other.data)
    , interpType(other.interpType)
    , edgeResolveType(other.edgeResolveType)
    , normalizeIntegers(normalizeIntegers)
    , normalizeCoordinates(normalizeCoordinates)
    , convertSRGB(convertSRGB)
{
    other.data = nullptr;
}

template<int D, class T>
Texture<D, T>& Texture<D, T>::operator=(Texture&& other)
{
    assert(this != &other);
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
        CUDA_CHECK(cudaDestroyTextureObject(this->texture));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }

    data = other.data;
    interpType = other.interpType;
    edgeResolveType = other.edgeResolveType;
    normalizeIntegers = other.normalizeIntegers;
    normalizeCoordinates = other.normalizeCoordinates;
    convertSRGB = other.convertSRGB;

    other.data = nullptr;

    return *this;
}

template<int D, class T>
Texture<D, T>::~Texture()
{
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
        CUDA_CHECK(cudaDestroyTextureObject(this->texture));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }
}

template<int D, class T>
void Texture<D, T>::Copy(const Byte* sourceData,
                         const TexDimType_t<D>& size,
                         const TexDimType_t<D>& offset,
                         int mipLevel)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = MakeCudaCopySize<D>(size);

    p.dstArray = levelArray;
    p.dstPos = MakeCudaOffset<D>(offset);

    p.srcPos = make_cudaPos(0, 0, 0);
    p.srcPtr = make_cudaPitchedPtr(const_cast<Byte*>(sourceData),
                                   p.extent.width * sizeof(T),
                                   p.extent.width, p.extent.height);
    CUDA_CHECK(cudaMemcpy3D(&p));
}

template<int D, class T>
GPUFence Texture<D, T>::CopyAsync(const Byte* sourceData,
                                  const TexDimType_t<D>& size,
                                  const TexDimType_t<D>& offset,
                                  int mipLevel,
                                  cudaStream_t stream)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = MakeCudaCopySize<D>(size);

    p.dstArray = levelArray;
    p.dstPos = MakeCudaOffset<D>(offset);

    p.srcPos = make_cudaPos(0, 0, 0);
    p.srcPtr = make_cudaPitchedPtr(const_cast<Byte*>(sourceData),
                                   p.extent.width * sizeof(T),
                                   p.extent.width, p.extent.height);

    CUDA_CHECK(cudaMemcpy3DAsync(&p, stream));
    return GPUFence(stream);
}

template<int D, class T>
Texture<D, T> Texture<D, T>::EmptyMipmappedTexture(uint32_t upToLevel) const
{
    uint32_t mipMax = std::numeric_limits<uint32_t>::min();
    for(int i = 0; i < D; i++)
    {
        uint32_t dimensionValue;
        if constexpr(D == 1) dimensionValue = this->dimensions;
        else dimensionValue = this->dimensions[i];

        dimensionValue = Utility::Log2Floor(dimensionValue);
        mipMax = std::max(dimensionValue, mipMax);
    }

    uint32_t mipCount = std::min(mipMax, upToLevel) + 1;
    Texture<D, T> t = Texture<D, T>(this->currentDevice,
                                    interpType,
                                    edgeResolveType,
                                    normalizeIntegers,
                                    normalizeCoordinates,
                                    convertSRGB,
                                    this->dimensions,
                                    mipCount);

    cudaArray_t dstLevelArray;
    cudaArray_t srcLevelArray;
    CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&srcLevelArray, data, 0));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&dstLevelArray, t.data, 0));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyDeviceToDevice;
    p.extent = MakeCudaCopySize<D>(this->dimensions);

    p.srcArray = srcLevelArray;
    p.dstArray = dstLevelArray;
    p.srcPos = make_cudaPos(0, 0, 0);
    p.dstPos = make_cudaPos(0, 0, 0);

    CUDA_CHECK(cudaMemcpy3D(&p));
    return t;
}

template<int D, class T>
CudaSurfaceRAII Texture<D, T>::GetMipLevelSurface(uint32_t level)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, level));
    // Construct surface object over it
    cudaResourceDesc rDesc;
    rDesc.resType = cudaResourceTypeArray;
    rDesc.res.array.array = levelArray;

    cudaSurfaceObject_t surface;
    CUDA_CHECK(cudaCreateSurfaceObject(&surface, &rDesc));

    return CudaSurfaceRAII(surface);
}

template<int D, class T>
InterpolationType Texture<D, T>::InterpType() const
{
    return interpType;
}

template<int D, class T>
EdgeResolveType Texture<D, T>::EdgeType() const
{
    return edgeResolveType;
}

template<int D, class T>
size_t Texture<D, T>::Size() const
{
    return TotalPixelCount<D>(this->dimensions) * sizeof(T);
}

template<int D, class T>
void Texture<D, T>::MigrateToOtherDevice(const CudaGPU*, cudaStream_t)
{
     // TODO: Implement texture migration
    assert(false);
}

template<int D, class T>
TextureArray<D, T>::TextureArray(const CudaGPU* device,
                                 InterpolationType interp,
                                 EdgeResolveType eResolve,
                                 bool normalizeIntegers,
                                 bool normalizeCoordinates,
                                 bool convertSRGB,
                                 const TexDimType_t<D>& size,
                                 unsigned int length,
                                 int mipCount)
    : TextureArrayI<D>(size, TextureChannelCount<T>::value, length, device)
    , interpType(interp)
    , edgeResolveType(eResolve)
{
    cudaExtent extent = MakeCudaExtent<D>(this->dimensions, length);
    cudaChannelFormatDesc d = cudaCreateChannelDesc<ChannelDescType_t<T>>();
    CUDA_CHECK(cudaSetDevice(device->DeviceId()));
    CUDA_MEMORY_CHECK(cudaMallocMipmappedArray(&data, &d, extent, mipCount,
                                               cudaArrayLayered));

    // Allocation Done now generate texture
    cudaResourceDesc rDesc = {};
    cudaTextureDesc tDesc = {};

    bool unormType = normalizeIntegers;

    rDesc.resType = cudaResourceType::cudaResourceTypeMipmappedArray;
    rDesc.res.mipmap.mipmap = data;

    tDesc.addressMode[0] = DetermineAddressMode(eResolve);
    tDesc.addressMode[1] = DetermineAddressMode(eResolve);
    tDesc.addressMode[2] = DetermineAddressMode(eResolve);
    tDesc.filterMode = DetermineFilterMode(interp);
    tDesc.readMode = (unormType) ? cudaReadModeNormalizedFloat : cudaReadModeElementType;

    tDesc.sRGB = convertSRGB;
    tDesc.borderColor[0] = 0.0f;
    tDesc.borderColor[1] = 0.0f;
    tDesc.borderColor[2] = 0.0f;
    tDesc.borderColor[3] = 0.0f;
    tDesc.normalizedCoords = normalizeCoordinates;
    tDesc.mipmapFilterMode = DetermineFilterMode(interp);

    //
    tDesc.maxAnisotropy = 4;
    tDesc.mipmapLevelBias = 0.0f;
    tDesc.minMipmapLevelClamp = -100.0f;
    tDesc.maxMipmapLevelClamp = 100.0f;

    CUDA_CHECK(cudaCreateTextureObject(&texture, &rDesc, &tDesc, nullptr));
}

template<int D, class T>
TextureArray<D, T>::TextureArray(TextureArray&& other)
    : TextureArrayI<D>(std::move(other))
    , data(other.data)
    , interpType(other.interpType)
    , edgeResolveType(other.edgeResolveType)
{
    other.data = nullptr;
}

template<int D, class T>
TextureArray<D, T>& TextureArray<D, T>::operator=(TextureArray&& other)
{
    assert(this != &other);
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
        CUDA_CHECK(cudaDestroyTextureObject(texture));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }

    data = other.data;
    interpType = other.interpType;
    edgeResolveType = other.edgeResolveType;

    other.data = nullptr;

    return *this;
}

template<int D, class T>
TextureArray<D, T>::~TextureArray()
{
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
        CUDA_CHECK(cudaDestroyTextureObject(texture));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }
}

template<int D, class T>
void TextureArray<D, T>::Copy(const Byte* sourceData,
                              const TexDimType_t<D>& size,
                              int layer,
                              const TexDimType_t<D>& offset,
                              int mipLevel)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = MakeCudaExtent<D>(size);

    p.dstArray = levelArray;
    p.dstPos = MakeCudaOffset<D>(offset, layer);

    p.srcPos = make_cudaPos(0, 0, 0);
    p.srcPtr = make_cudaPitchedPtr(const_cast<Byte*>(sourceData),
                                   p.extent.width * sizeof(T),
                                   p.extent.width, p.extent.height);

    CUDA_CHECK(cudaMemcpy3D(&p));
}

template<int D, class T>
GPUFence TextureArray<D, T>::CopyAsync(const Byte* sourceData,
                                       const TexDimType_t<D>& size,
                                       int layer,
                                       const TexDimType_t<D>& offset,
                                       int mipLevel,
                                       cudaStream_t stream)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = MakeCudaExtent<D>(size);

    p.dstArray = levelArray;
    p.dstPos = MakeCudaOffset<D>(offset, layer);

    p.srcPos = make_cudaPos(0, 0, 0);
    p.srcPtr = make_cudaPitchedPtr(const_cast<Byte*>(sourceData),
                                   p.extent.width * sizeof(T),
                                   p.extent.width, p.extent.height);

    CUDA_CHECK(cudaMemcpy3DAsync(&p, stream));
    return GPUFence(stream);
}

template<int D, class T>
InterpolationType TextureArray<D, T>::InterpType() const
{
    return interpType;
}

template<int D, class T>
EdgeResolveType TextureArray<D, T>::EdgeType() const
{
    return edgeResolveType;
}

template<int D, class T>
size_t TextureArray<D, T>::Size() const
{
    return TotalPixelCount<D>(this->dimensions) * this->length * sizeof(T);
}

template<int D, class T>
void TextureArray<D, T>::MigrateToOtherDevice(const CudaGPU*, cudaStream_t)
{
     // TODO: Implement texture migration
    assert(false);
}

template <class T>
TextureCube<T>::TextureCube(const CudaGPU* device,
                            InterpolationType interp,
                            EdgeResolveType eResolve,
                            bool normalizeIntegers,
                            bool convertSRGB,
                            const Vector2ui& size,
                            int mipCount)
    : TextureCubeI(size, TextureChannelCount<T>::value, device)
    , interpType(interp)
    , edgeResolveType(eResolve)
{
    assert(this->dimensions[0] == this->dimensions[1]);
    cudaExtent extent = make_cudaExtent(this->dimensions[0], this->dimensions[1], CUBE_FACE_COUNT);
    cudaChannelFormatDesc d = cudaCreateChannelDesc<ChannelDescType_t<T>>();
    CUDA_CHECK(cudaSetDevice(device->DeviceId()));
    CUDA_MEMORY_CHECK(cudaMallocMipmappedArray(&data, &d, extent, mipCount,
                                               cudaArrayCubemap));

    // Allocation Done now generate texture
    cudaResourceDesc rDesc = {};
    cudaTextureDesc tDesc = {};

    bool unormType = normalizeIntegers;

    rDesc.resType = cudaResourceType::cudaResourceTypeMipmappedArray;
    rDesc.res.mipmap.mipmap = data;

    tDesc.addressMode[0] = DetermineAddressMode(eResolve);
    tDesc.addressMode[1] = DetermineAddressMode(eResolve);
    tDesc.addressMode[2] = DetermineAddressMode(eResolve);
    tDesc.filterMode = DetermineFilterMode(interp);
    tDesc.readMode = (unormType) ? cudaReadModeNormalizedFloat : cudaReadModeElementType;

    tDesc.sRGB = convertSRGB;
    tDesc.borderColor[0] = 0.0f;
    tDesc.borderColor[1] = 0.0f;
    tDesc.borderColor[2] = 0.0f;
    tDesc.borderColor[3] = 0.0f;
    tDesc.normalizedCoords = 1;
    tDesc.mipmapFilterMode = DetermineFilterMode(interp);

    //
    tDesc.maxAnisotropy = 4;
    tDesc.mipmapLevelBias = 0.0f;
    tDesc.minMipmapLevelClamp = -100.0f;
    tDesc.maxMipmapLevelClamp = 100.0f;

    CUDA_CHECK(cudaCreateTextureObject(&texture, &rDesc, &tDesc, nullptr));
}

template<class T>
TextureCube<T>::TextureCube(TextureCube&& other)
    : TextureCubeI(std::move(other))
    , data(other.data)
    , interpType(other.interpType)
    , edgeResolveType(other.edgeResolveType)
{
    other.data = nullptr;
}

template<class T>
TextureCube<T>& TextureCube<T>::operator=(TextureCube&& other)
{
    assert(this != &other);
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
        CUDA_CHECK(cudaDestroyTextureObject(texture));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }

    data = other.data;
    interpType = other.interpType;
    edgeResolveType = other.edgeResolveType;

    other.data = nullptr;
    return *this;
}

template <class T>
TextureCube<T>::~TextureCube()
{
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
        CUDA_CHECK(cudaDestroyTextureObject(texture));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }
}

template <class T>
void TextureCube<T>::Copy(const Byte* sourceData,
                          const Vector2ui& size,
                          CubeTexSide side,
                          const Vector2ui&,
                          int mipLevel)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    size_t sideIndex = static_cast<size_t>(side);

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = make_cudaExtent(size[0], size[1], 0);

    p.dstArray = levelArray;
    p.dstPos = make_cudaPos(size[0], size[1], sideIndex);

    p.srcPos = make_cudaPos(0, 0, 0);
    p.srcPtr = make_cudaPitchedPtr(const_cast<Byte*>(sourceData),
                                   p.extent.width * sizeof(T),
                                   p.extent.width, p.extent.height);

    CUDA_CHECK(cudaMemcpy3D(&p));
}

template <class T>
GPUFence TextureCube<T>::CopyAsync(const Byte* sourceData,
                                   const Vector2ui& size,
                                   CubeTexSide side,
                                   const Vector2ui&,
                                   int mipLevel,
                                   cudaStream_t stream)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaSetDevice(this->currentDevice->DeviceId()));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    size_t sideIndex = static_cast<size_t>(side);

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = make_cudaExtent(size[0], size[1], 0);

    p.dstArray = levelArray;
    p.dstPos = make_cudaPos(size[0], size[1], sideIndex);

    p.srcPos = make_cudaPos(0, 0, 0);
    p.srcPtr = make_cudaPitchedPtr(const_cast<Byte*>(sourceData),
                                   p.extent.width * sizeof(T),
                                   p.extent.width, p.extent.height);

    CUDA_CHECK(cudaMemcpy3DAsync(&p, stream));
    return GPUFence(stream);
}

template <class T>
InterpolationType TextureCube<T>::InterpType() const
{
    return interpType;
}

template <class T>
EdgeResolveType TextureCube<T>::EdgeType() const
{
    return edgeResolveType;
}

template<class T>
size_t TextureCube<T>::Size() const
{
    return static_cast<size_t>(this->dimensions[0]) * this->dimensions[1] * CUBE_FACE_COUNT * sizeof(T);
}

template <class T>
void TextureCube<T>::MigrateToOtherDevice(const CudaGPU*, cudaStream_t)
{
    // TODO: Implement texture migration
    assert(false);
}