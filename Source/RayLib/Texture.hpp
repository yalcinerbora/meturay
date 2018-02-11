
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
