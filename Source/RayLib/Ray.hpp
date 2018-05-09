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
	return direction;
}

template<class T>
__device__ __host__
inline const Vector<3, T>& Ray<T>::getPosition() const
{
	return position;
}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsSphere(Vector<3, T>& pos, float& t,
									 const Vector<3, T>& sphereCenter,
									 float sphereRadius) const
{
	// Geometric solution
	Vector<3, T> centerDir = sphereCenter - position;
	T beamCenterDistance = centerDir.DotProduct(direction);
	T beamNormalLengthSqr = centerDir.DotProduct(centerDir) -
							beamCenterDistance * beamCenterDistance;
	float beamHalfLengthSqr = sphereRadius * sphereRadius - beamNormalLengthSqr;
	if(beamHalfLengthSqr > 0.0f)
	{
		// Inside Square
		float beamHalfLength = std::sqrt(beamHalfLengthSqr);
		float t0 = beamCenterDistance - beamHalfLength;
		float t1 = beamCenterDistance + beamHalfLength;
		if(t1 >= 0.0f)
		{
			t = (t0 >= 0.0f) ? t0 : t1;
			intersectPos = position + t * normDir;
			return true;
		}
	}
	return false;
}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsTriangle(Vector<3, T>& baryCoords, float& t,									   
									   const Vector<3, T> triCorners[3],
									   bool cullFace) const
{
	return IntersectsTriangle(baryCoords, t,
							  triCorners[0],
							  triCorners[1],
							  triCorners[2]);
}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsTriangle(Vector<3, T>& baryCoords, float& t,									   
									   const Vector<3, T>& t0,
									   const Vector<3, T>& t1,
									   const Vector<3, T>& t2,
									   bool cullFace) const
{
	// Matrix Solution
	// Kramers Rule
	Vector<3, T> abDiff = t0 - t1;
	Vector<3, T> acDiff = t0 - t2;
	Vector<3, T> aoDiff = t0 - position;

	if(cullFace)
	{
		// TODO this is wrong??
		IEVector3 normal = abDiff.CrossProduct(acDiff).Normalize();
		T cos = direction.DotProduct(normal);
		if(cos > 0) return false;
	}

	Matrix<3, T> A = Matrix<3, T>(abDiff, acDiff, direction);
	Matrix<3, T> betaA = Matrix<3, T>(aoDiff, acDiff, direction);
	Matrix<3, T> gammaA = Matrix<3, T>(abDiff, aoDiff, direction);
	Matrix<3, T> tA = Matrix<3, T>(abDiff, acDiff, aoDiff);

	T aDetInv = 1.0f / A.Determinant();
	T beta = betaA.Determinant() * aDetInv;
	T gamma = gammaA.Determinant() * aDetInv;
	T alpha = 1.0f - beta - gamma;
	T rayT = tA.Determinant() * aDetInv;

	if(beta >= 0.0f && beta <= 1.0f &&
	   gamma >= 0.0f && gamma <= 1.0f &&
	   alpha >= 0.0f && alpha <= 1.0f &&
	   rayT >= 0.0f)
	{
		baryCoords = Vector<3, T>(alpha, beta, gamma);
		t = rayT;
	}
	else
		return false;
	return true;
}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsAABB(const Vector<3, T>& min,
								   const Vector<3, T>& max) const
{
	Vector3 invD = Vector3(1.0f) / direction;
	Vector3 t0 = (min - position) * invD;
	Vector3 t1 = (max - position) * invD;

	float tMin = -std::numeric_limits<float>::max();
	float tMax = std::numeric_limits<float>::max();

	#pragma unroll
	for(int i = 0; i < 3; i++)
	{
		tMin = std::max(tMin, std::min(t0[i], t1[i]));
		tMax = std::min(tMax, std::max(t0[i], t1[i]));
	}
	return tMax >= tMin;
}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsAABB(Vector<3, T>& pos, float& t,
								   const Vector<3, T>& min,
								   const Vector<3, T>& max) const
{
	Vector3 invD = Vector3(1.0f) / direction;
	Vector3 t0 = (min - position) * invD;
	Vector3 t1 = (max - position) * invD;

	float tMin = -std::numeric_limits<float>::max();
	float tMax = std::numeric_limits<float>::max();
	float t = std::numeric_limits<float>::max();

	#pragma unroll
	for(int i = 0; i < 3; i++)
	{
		tMin = std::max(tMin, std::min(t0[i], t1[i]));
		tMax = std::min(tMax, std::max(t0[i], t1[i]));
		t = (t0[i] > 0.0f) ? std::min(t, t0[i]) : t;
		t = (t1[i] > 0.0f) ? std::min(t, t1[i]) : t;
	}
	pos = position + t * direction;
	return (tMax >= tMin) ? t : FLT_MAX;	
}

template<class T>
__device__ __host__ 
inline Ray<T> Ray<T>::Reflect(const Vector<3, T>& normal) const
{
	Vector<3, T> nDir = -direction;
	nDir = 2.0f * nDir.DotProduct(normal) * normal - nDir;
	return Ray(nDir, position);
}

template<class T>
__device__ __host__ 
inline Ray<T>& Ray<T>::ReflectSelf(const Vector<3, T>& normal)
{
	Vector<3, T> nDir = -direction;
	direction = (2.0f * nDir.DotProduct(normal) * normal - nDir) * length;
	return *this;
}

template<class T>
__device__ __host__
inline bool Ray<T>::Refract(Ray& out, const Vector<3, T>& normal,
							T fromMedium, T toMedium) const
{
	T cosTetha = -normal.DotProduct(direction);
	T indexRatio = fromMedium / toMedium;

	float delta = 1.0f - indexRatio * indexRatio * (1.0f - cosTetha * cosTetha);
	if(delta > 0.0f)
	{
		out.direction = indexRatio * direction + normal * (cosTetha * indexRatio - std::sqrt(delta));
		out.position = position;
		return true;
	}
	return false;
}

template<class T>
__device__ __host__
inline bool Ray<T>::RefractSelf(const Vector<3, T>& normal,
								T fromMedium, T toMedium)
{
	T cosTetha = -normal.DotProduct(direction);
	T indexRatio = fromMedium / toMedium;

	T delta = 1.0f - indexRatio * indexRatio * (1.0f - cosTetha * cosTetha);
	if(delta > 0.0f)
	{
		direction = indexRatio * direction + normal * (cosTetha * indexRatio - std::sqrt(delta));
		return true;
	}
	return false;
}

template<class T>
__device__ __host__
inline Ray<T> Ray<T>::RandomRayCosine(float xi0, float xi1,
									  const Vector<3, T>& normal,
									  const Vector<3, T>& position)
{
	Vector<3, T> randomDir;
	randomDir[0] = std::sqrt(xi0) * std::cos(2.0f * MathConstants::PI * xi1);
	randomDir[1] = std::sqrt(xi0) * std::sin(2.0f * MathConstants::PI * xi1);
	randomDir[2] = std::sqrt(1.0f - xi0);

	Quaternion<T> q = QuatF::RotationBetweenZAxis(normal);
	Vector<3, T> rotatedDir = q.ApplyRotation(randomDir);
	return Ray(rotatedDir, position);
}

template<class T>
__device__ __host__
inline Ray<T> Ray<T>::RandomRayUnfirom(float xi0, float xi1,
									   const Vector<3, T>& normal,
									   const Vector<3, T>& position)
{
	Vector<3, T> randomDir;
	randomDir[0] = std::sqrt(1.0f - xi0 * xi0) * std::cos(2.0f * MathConstants::PI * xi1);
	randomDir[1] = std::sqrt(1.0f - xi0 * xi0) * std::sin(2.0f * MathConstants::PI * xi1);
	randomDir[2] = xi0;

	Quaternion<T> q = QuatF::RotationBetweenZAxis(normal);
	randomDir = q.ApplyRotation(randomDir);
	return Ray(randomDir, position);
}

template<class T>
__device__ __host__ 
inline Ray<T> Ray<T>::NormalizeDir() const
{
	Ray(direction.Normalize(), position);
}

template<class T>
__device__ __host__ 
inline Ray<T>& Ray<T>::NormalizeDirSelf()
{
	direction.NormalizeSelf();
}

template<class T>
__device__ __host__ 
inline Ray<T> Ray<T>::Advance(T t) const
{
	return Ray(direction, position + t * direction);
}

template<class T>
__device__ __host__ 
inline Ray<T>& Ray<T>::AdvanceSelf(T t)
{
	position += t * direction;
	return *this;
}

template<class T>
__device__ __host__ 
inline Ray<T> Ray<T>::Transform(const Matrix<4, T>& mat) const
{
	return Ray((mat * Vector4(direction, 0.0f)).Normalize(),
			   mat * Vector4(position, 1.0f));
}

template<class T>
__device__ __host__ 
inline Ray<T>& Ray<T>::TransformSelf(const Matrix<4, T>&)
{
	direction = (mat * Vector<4, T>(direction, 0.0f)).Normalize();
	position = mat * Vector<4, T>(position, 1.0f);
	return *this;
}

template<class T>
__device__ __host__ 
inline Vector<3, T> Ray<T>::AdvancedPos(T t) const
{
	return position + t * direction;
}
