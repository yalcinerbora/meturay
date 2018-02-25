#pragma once

template<class T>
__device__ __host__
inline constexpr Ray<T>::Ray(const Vector<3, T>& direction, const Vector<3, T>& position)
	: direction(direction)
	, position(position)
{}

template<class T>
__device__ __host__
inline constexpr Ray<T>::Ray(const Vector3 vec[2])
	: direction(vec[0])
	, position(vec[1])
{}

template<class T>
__device__ __host__ 
inline Ray<T>& Ray<T>::operator=(const Vector3 vec[2])
{
	direction = vec[0];
	position = vec[1];
}

template<class T>
__device__ __host__ 
inline const Vector<3, T>& Ray<T>::getDirection() const
{
	return directon;
}

template<class T>
__device__ __host__
inline const Vector<3, T>& Ray<T>::getPosition() const
{
	return position;
}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsSphere(Vector<3, T>& pos,
									 float& t,
									 const Vector<3, T>& sphereCenter,
									 float sphereRadius) const
{

}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsTriangle(Vector<3, T>& baryCoords, float& t,									   
									   const Vector<3, T> triCorners[3],
									   bool cullFace) const
{

}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsTriangle(Vector<3, T>& baryCoords, float& t,									   
									   const Vector<3, T>& t0,
									   const Vector<3, T>& t1,
									   const Vector<3, T>& t2,
									   bool cullFace) const
{

}


template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsAABB(const Vector<3, T>& min,
								   const Vector<3, T>& max) const
{

}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsAABB(Vector<3, T>& pos, float& t,
								   const Vector<3, T>& min,
								   const Vector<3, T>& max) const
{

}

template<class T>
__device__ __host__ 
inline Ray<T> Ray<T>::Reflect(const Vector<3, T>& normal) const
{

}

template<class T>
__device__ __host__ 
inline Ray<T>& Ray<T>::ReflectSelf(const Vector<3, T>& normal)
{

}

template<class T>
__device__ __host__
inline bool Ray<T>::Refract(Ray& out, const Vector<3, T>& normal,
							float fromMedium, float toMedium) const
{

}

template<class T>
__device__ __host__
inline bool Ray<T>::RefractSelf(const Vector<3, T>& normal,
								float fromMedium, float toMedium)
{

}

template<class T>
__device__ __host__
inline Ray<T> Ray<T>::RandomRayCosine(float xi0, float xi1,
									  const Vector<3, T>& normal,
									  const Vector<3, T>& position)
{

}

template<class T>
__device__ __host__
inline Ray<T> Ray<T>::RandomRayUnfirom(float xi0, float xi1,
									   const Vector<3, T>& normal,
									   const Vector<3, T>& position)
{

}

template<class T>
__device__ __host__ 
inline Ray<T> Ray<T>::NormalizeDir() const
{

}

template<class T>
__device__ __host__ 
inline Ray<T>& Ray<T>::NormalizeDirSelf()
{

}

template<class T>
__device__ __host__ 
inline Ray<T> Ray<T>::Advance(float) const
{

}

template<class T>
__device__ __host__ 
inline Ray<T>& Ray<T>::AdvanceSelf(float)
{

}

template<class T>
__device__ __host__ 
inline Ray<T> Ray<T>::Transform(const Matrix4x4&) const
{

}

template<class T>
__device__ __host__ 
inline Ray<T>& Ray<T>::TransformSelf(const Matrix4x4&)
{

}

template<class T>
__device__ __host__ 
inline Vector<3, T> Ray<T>::AdvancedPos(float t) const
{

}
