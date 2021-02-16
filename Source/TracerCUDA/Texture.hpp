
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

template<int D, int C>
TextureI<D, C>::TextureI(TextureI&& other)
    : texture(other.texture)
{
    other.texture = 0;
}

template<int D, int C>
TextureI<D, C>& TextureI<D, C>::operator=(TextureI&& other)
{
    assert(this != &other);
    texture = other.texture;
    other.texture = 0;
}

template<int D, int C>
TextureArrayI<D, C>::TextureArrayI(TextureArrayI&& other)
    : texture(other.texture)
    , length(other.length)
{
    other.texture = 0;
    other.length = 0;
}

template<int D, int C>
TextureArrayI<D, C>& TextureArrayI<D, C>::operator=(TextureArrayI&& other)
{
    assert(this != &other);
    texture = other.texture;
    length = other.length;

    other.texture = 0;
    other.length = 0;
}


template<int C>
TextureCubeI<C>::TextureCubeI(TextureCubeI&& other)
    : texture(other.texture)
{
    other.texture = 0;
}

template<int C>
TextureCubeI<C>& TextureCubeI<C>::operator=(TextureCubeI&& other)
{
    assert(this != &other);
    texture = other.texture;
    other.texture = 0;
}

template<int D, class T>
Texture<D, T>::Texture(int deviceId,
                       InterpolationType interp,
                       EdgeResolveType eResolve,
                       bool normalizeIntegers,
                       bool normalizeCoordinates,
                       bool convertSRGB,
                       const TexDimType_t<D>& dim,
                       int mipCount)
    : DeviceLocalMemoryI(deviceId)
    , TextureI<D, TextureChannelCount<T>::value>(texture)
    , dim(dim)
    , interpType(interp)
    , edgeResolveType(eResolve)
{
    cudaExtent extent = MakeCudaExtent<D>(dim);
    cudaChannelFormatDesc d = cudaCreateChannelDesc<T>();
    CUDA_CHECK(cudaSetDevice(deviceId));
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

    //
    tDesc.maxAnisotropy = 4;
    tDesc.mipmapLevelBias = 0.0f;
    tDesc.minMipmapLevelClamp = -100.0f;
    tDesc.maxMipmapLevelClamp = 100.0f;

    CUDA_CHECK(cudaCreateTextureObject(&texture, &rDesc, &tDesc, nullptr));
}

template<int D, class T>
Texture<D, T>::Texture(Texture&& other)
    : DeviceLocalMemoryI(other)
    , TextureI<D, TextureChannelCount<T>::value>(std::move(other))
    , data(other.data)
    , dim(other.dim)
    , interpType(other.interpType)
    , edgeResolveType(other.edgeResolveType)
{
    other.dim = TexDimType<D>::ZERO;
    other.data = nullptr;
}

template<int D, class T>
Texture<D, T>& Texture<D, T>::operator=(Texture&& other)
{
    assert(this != &other);
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(currentDevice));
        CUDA_CHECK(cudaDestroyTextureObject(texture));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }

    data = other.data;
    dim = other.dim;
    interpType = other.interpType;
    edgeResolveType = other.edgeResolveType;

    other.data = nullptr;
    other.dim = TexDimType<D>::ZERO;

    return *this;
}

template<int D, class T>
Texture<D, T>::~Texture()
{
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(currentDevice));
        CUDA_CHECK(cudaDestroyTextureObject(texture));
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
    CUDA_CHECK(cudaSetDevice(currentDevice));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = MakeCudaExtent<D>(size);
    
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
    CUDA_CHECK(cudaSetDevice(currentDevice));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    p.extent = MakeCudaExtent<D>(size);

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
const TexDimType_t<D>& Texture<D, T>::Dim() const
{
    return dim;
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
    return TotalPixelCount<D>(dim) * sizeof(T);
}

template<int D, class T>
void Texture<D, T>::MigrateToOtherDevice(int deviceTo, cudaStream_t stream)
{
     // TODO: Implement texture migration
    assert(false);
}

template<int D, class T>
TextureArray<D, T>::TextureArray(int deviceId,
                                 InterpolationType interp,
                                 EdgeResolveType eResolve,
                                 bool normalizeIntegers,
                                 bool normalizeCoordinates,
                                 bool convertSRGB,
                                 const TexDimType_t<D>& dim,
                                 unsigned int length,
                                 int mipCount)
    : DeviceLocalMemoryI(deviceId)
    , TextureArrayI<D, TextureChannelCount<T>::value>(texture, length)
    , dim(dim)
    , length(length)
    , interpType(interp)
    , edgeResolveType(eResolve)
{
    cudaExtent extent = MakeCudaExtent<D>(dim, length);   
    cudaChannelFormatDesc d = cudaCreateChannelDesc<T>();
    CUDA_CHECK(cudaSetDevice(deviceId));
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
    : DeviceLocalMemoryI(other)
    , TextureArrayI<D, TextureChannelCount<T>::value>(std::move(other))
    , data(other.data)
    , dim(other.dim)
    , interpType(other.interpType)
    , edgeResolveType(other.edgeResolveType)
{
    other.dim = TexDimType<D>::ZERO;
    other.data = nullptr;
}

template<int D, class T>
TextureArray<D, T>& TextureArray<D, T>::operator=(TextureArray&& other)
{
    assert(this != &other);
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(currentDevice));
        CUDA_CHECK(cudaDestroyTextureObject(texture));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }

    data = other.data;
    dim = other.dim;
    interpType = other.interpType;
    edgeResolveType = other.edgeResolveType;

    other.data = nullptr;
    other.dim = TexDimType<D>::ZERO;

    return *this;
}

template<int D, class T>
TextureArray<D, T>::~TextureArray()
{
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(currentDevice));
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
    CUDA_CHECK(cudaSetDevice(currentDevice));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    cudaExtent extent = MakeCudaExtent<D>(size);

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
    CUDA_CHECK(cudaSetDevice(currentDevice));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    cudaExtent extent = MakeCudaExtent<D>(size);

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
const TexDimType_t<D>& TextureArray<D, T>::Dim() const
{
    return dim;
}

template<int D, class T>
unsigned int TextureArray<D, T>::Length() const
{
    return length;
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
    return TotalPixelCount<D>(dim) * length * sizeof(T);
}

template<int D, class T>
void TextureArray<D, T>::MigrateToOtherDevice(int deviceTo, cudaStream_t stream)
{
     // TODO: Implement texture migration
    assert(false);
}

template <class T>
TextureCube<T>::TextureCube(int deviceId,
                            InterpolationType interp,
                            EdgeResolveType eResolve,
                            bool normalizeIntegers,
                            bool convertSRGB,
                            const Vector2ui& dim,
                            int mipCount)
    : DeviceLocalMemoryI(deviceId)
    , TextureCubeI<TextureChannelCount<T>::value>(texture)
    , dim(dim)
    , interpType(interp)
    , edgeResolveType(eResolve)
{
    assert(dim[0] == dim[1]);
    cudaExtent extent = make_cudaExtent(dim[0], dim[1], CUBE_FACE_COUNT);
    cudaChannelFormatDesc d = cudaCreateChannelDesc<T>();
    CUDA_CHECK(cudaSetDevice(deviceId));
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
    : DeviceLocalMemoryI(other)
    , TextureCubeI<TextureChannelCount<T>::value>(std::move(other))
    , data(other.data)
    , dim(other.dim)
    , interpType(other.interpType)
    , edgeResolveType(other.edgeResolveType)
{
    other.dim = Zero2ui;
    other.data = nullptr;
}

template<class T>
TextureCube<T>& TextureCube<T>::operator=(TextureCube&& other)
{
    assert(this != &other);
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(currentDevice));
        CUDA_CHECK(cudaDestroyTextureObject(texture));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }

    data = other.data;
    dim = other.dim;
    interpType = other.interpType;
    edgeResolveType = other.edgeResolveType;

    other.data = nullptr;
    other.dim = Zero2ui;

    return *this;
}

template <class T>
TextureCube<T>::~TextureCube()
{
    if(data)
    {
        CUDA_CHECK(cudaSetDevice(currentDevice));
        CUDA_CHECK(cudaDestroyTextureObject(texture));
        CUDA_CHECK(cudaFreeMipmappedArray(data));
    }
}

template <class T>
void TextureCube<T>::Copy(const Byte* sourceData,
                          const Vector2ui& size,
                          CubeTexSide side,
                          const Vector2ui& offset,
                          int mipLevel)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaSetDevice(currentDevice));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    size_t sideIndex = static_cast<size_t>(side);

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    cudaExtent extent = make_cudaExtent(size[0], size[1], 0);

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
                                   const Vector2ui& offset,                                   
                                   int mipLevel,
                                   cudaStream_t stream)
{
    cudaArray_t levelArray;
    CUDA_CHECK(cudaSetDevice(currentDevice));
    CUDA_CHECK(cudaGetMipmappedArrayLevel(&levelArray, data, mipLevel));

    size_t sideIndex = static_cast<size_t>(side);

    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    cudaExtent extent = make_cudaExtent(size[0], size[1], 0);

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
const Vector2ui& TextureCube<T>::Dim() const
{
    return dim;
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
    return static_cast<size_t>(dim[0]) * dim[1] * CUBE_FACE_COUNT * sizeof(T);
}

template <class T>
void TextureCube<T>::MigrateToOtherDevice(int deviceTo, cudaStream_t stream)
{
    // TODO: Implement texture migration
    assert(false);
}