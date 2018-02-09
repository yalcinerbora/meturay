
template<int D>
__device__ __host__ int64_t Texture<D>::DimToLinear(int i, int j, int k);

template<int D>
__device__ __host__ int64_t Texture<D>::DimToLinear(int i, int j, int k)
{

}

template<int D>
__device__ __host__ Texture<D>::Texture(const TextureData& t,
										const Sampler&)
{

}

template<int D>
template<class T>
__device__ __host__ const T Texture<D>::operator()(const Vector<D, int>&) const
{

}
//
//template<int D>
//template<class T>
//__device__ __host__ const T Texture<D>::operator()(const Vector<D + 1, int>&) const
//{
//
//}
//
//template<int D>
//template<class T>
//__device__ __host__ const T Texture<D>::operator()(const Vector<D, float>&) const
//{
//
//}
//
//template<int D>
//template<class T>
//__device__ __host__ const T Texture<D>::operator()(const Vector<D + 1, float>&) const
//{
//
//}
