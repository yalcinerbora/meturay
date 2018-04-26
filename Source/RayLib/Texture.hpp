
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
__host__ Texture<D, T>::Texture(InterpolationType interp,
								EdgeResolveType eResolve,
								bool unormType,
								const Vector<D, unsigned int>& dim)
{
	cudaExtent extent = {};
	if(D == 1)
	{
		extent = make_cudaExtent(dim[0], 0 , 0);
	}
	else if(D == 2)
	{
		extent = make_cudaExtent(dim[0], dim[1], 0);
	}
	else if(D == 3)
	{
		extent = make_cudaExtent(dim[0], dim[1], dim[2]);
	}

	cudaChannelFormatDesc d = cudaCreateChannelDesc<T>();
	CUDA_CHECK(cudaMallocMipmappedArray(&data, &d, extent, 1));
	
	// Allocation Done now generate texture
	cudaResourceDesc rDesc = {};
	cudaTextureDesc tDesc = {};

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
__device__ const T Texture<D, T>::operator()(const Vector<D, float>& index) const
{
	// TODO: change to "if constexpr(statement)" when cuda supports it
	if(D == 1)
	{
		return tex1D<T>(t, index[0]);
	}
	else if(D == 2)
	{
		return tex2D<T>(t, index[0], index[1]);
	}
	else
	{
		return tex3D<T>(t, index[0], index[1], index[2]);
	}
}

template<int D, class T>
__device__ const T Texture<D, T>::operator()(const Vector<D + 1, float>& index) const
{
	// TODO: change to "if constexpr(statement)" when cuda supports it
	if(D == 1)
	{
		return tex1DLod<T>(t, index[0], index[1]);
	}
	else if(D == 2)
	{
		return tex2DLod<T>(t, index[0], index[1], index[2]);
	}
	else
	{
		return tex3DLod<T>(t, index[0], index[1], index[2], index[3]);
	}
}


template<int D, class T>
void Texture<D, T>::Copy(const byte* sourceData,
						 const Vector<D, unsigned int>& size,
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
	p.srcPtr = make_cudaPitchedPtr(const_cast<byte*>(sourceData),
								   size[0] * sizeof(T),
								   size[0], size[1]);
	
	CUDA_CHECK(cudaMemcpy3D(&p));
}

template<int D, class T>
void Texture<D, T>::MigrateToOtherDevice(int deviceTo, cudaStream_t stream)
{

}
