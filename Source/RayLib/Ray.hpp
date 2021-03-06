#pragma once

template<class T>
__device__ __host__
inline constexpr Ray<T>::Ray(const Vector<3, T>& direction, const Vector<3, T>& position)
    : direction(direction)
    , position(position)
{}

template<class T>
__device__ __host__
inline constexpr Ray<T>::Ray(const Vector<3, T> vec[2])
    : direction(vec[0])
    , position(vec[1])
{}

template<class T>
__device__ __host__
inline Ray<T>& Ray<T>::operator=(const Vector<3, T> vec[2])
{
    direction = vec[0];
    position = vec[1];
    return *this;
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
inline bool Ray<T>::IntersectsSphere(Vector<3, T>& intersectPos, T& t,
                                     const Vector<3, T>& sphereCenter,
                                     T sphereRadius) const
{
    // Geometric solution
    Vector<3, T> centerDir = sphereCenter - position;
    T beamCenterDistance = centerDir.Dot(direction);
    T beamNormalLengthSqr = centerDir.Dot(centerDir) -
                            beamCenterDistance * beamCenterDistance;
    T beamHalfLengthSqr = sphereRadius * sphereRadius - beamNormalLengthSqr;
    if(beamHalfLengthSqr > 0.0f)
    {
        // Inside Square
        T beamHalfLength = sqrt(beamHalfLengthSqr);
        T t0 = beamCenterDistance - beamHalfLength;
        T t1 = beamCenterDistance + beamHalfLength;
        if(t1 >= 0.0f)
        {
            t = (t0 >= 0.0f) ? t0 : t1;
            intersectPos = position + t * direction;
            return true;
        }
    }
    return false;
}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                       const Vector<3, T> triCorners[3],
                                       bool cullFace) const
{
    return IntersectsTriangle(baryCoords, t,
                              triCorners[0],
                              triCorners[1],
                              triCorners[2],
                              cullFace);
}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
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
        Vector<3, T> normal = Cross(abDiff, acDiff).Normalize();
        T cos = direction.Dot(normal);
        if(cos > 0) return false;
    }

    Vector<3, T> aData[] = {abDiff, acDiff, direction};
    Vector<3, T> betaAData[] = {aoDiff, acDiff, direction};
    Vector<3, T> gammaAData[] = {abDiff, aoDiff, direction};
    Vector<3, T> tAData[] = {abDiff, acDiff, aoDiff};

    Matrix<3, T> A = Matrix<3, T>(aData);
    Matrix<3, T> betaA = Matrix<3, T>(betaAData);
    Matrix<3, T> gammaA = Matrix<3, T>(gammaAData);
    Matrix<3, T> tA = Matrix<3, T>(tAData);

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
        return true;
    }
    else return false;
}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsAABB(const Vector<3, T>& aabbMin,
                                   const Vector<3, T>& aabbMax,
                                   const Vector<2, T>& tMinMax) const
{
    Vector<3,T> invD = Vector<3, T>(1) / direction;
    Vector<3,T> t0 = (aabbMin - position) * invD;
    Vector<3,T> t1 = (aabbMax - position) * invD;

    T tMin = tMinMax[0];
    T tMax = tMinMax[1];

    UNROLL_LOOP
    for(int i = 0; i < 3; i++)
    {
        if(invD[i] < 0) HybridFuncs::Swap(t0[i], t1[i]);

        tMin = max(tMin, min(t0[i], t1[i]));
        tMax = min(tMax, max(t0[i], t1[i]));
    }
    return tMax >= tMin;
}

template<class T>
__device__ __host__
inline bool Ray<T>::IntersectsAABB(Vector<3, T>& pos, T& tOut,
                                   const Vector<3, T>& aabbMin,
                                   const Vector<3, T>& aabbMax,
                                   const Vector<2, T>& tMinMax) const
{
    Vector<3,T> invD = Vector<3, T>(1) / direction;
    Vector<3,T> t0 = (aabbMin - position) * invD;
    Vector<3,T> t1 = (aabbMax - position) * invD;

    T tMin = tMinMax[0];
    T tMax = tMinMax[1];
    T t = tMin;

    UNROLL_LOOP
    for(int i = 0; i < 3; i++)
    {
        if(invD[i] < 0) HybridFuncs::Swap(t0[i], t1[i]);

        tMin = max(tMin, min(t0[i], t1[i]));
        tMax = min(tMax, max(t0[i], t1[i]));

        t = (t0[i] > 0.0f) ? min(t, t0[i]) : t;
        t = (t1[i] > 0.0f) ? min(t, t1[i]) : t;
    }

    // Calculate intersect position and the multiplier t
    if(tMax >= tMin)
    {
        tOut = t;
        pos = position + t * direction;
    }
    return (tMax >= tMin);
}

template<class T>
__device__ __host__
inline Ray<T> Ray<T>::Reflect(const Vector<3, T>& normal) const
{
    Vector<3, T> nDir = -direction;
    nDir = static_cast<T>(2.0) * nDir.Dot(normal) * normal - nDir;
    return Ray(nDir, position);
}

template<class T>
__device__ __host__
inline Ray<T>& Ray<T>::ReflectSelf(const Vector<3, T>& normal)
{
    Vector<3, T> nDir = -direction;
    direction = (static_cast<T>(2.0) * nDir.Dot(normal) * normal - nDir);
    return *this;
}

template<class T>
__device__ __host__
inline bool Ray<T>::Refract(Ray& out, const Vector<3, T>& normal,
                            T fromMedium, T toMedium) const
{
    T cosTetha = -normal.Dot(direction);
    T indexRatio = fromMedium / toMedium;

    T delta = static_cast<T>(1.0) - indexRatio * indexRatio * (static_cast<T>(1.0) - cosTetha * cosTetha);
    if(delta > static_cast<T>(0.0))
    {
        out.direction = indexRatio * direction + normal * (cosTetha * indexRatio - sqrt(delta));
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
    T cosTetha = -normal.Dot(direction);
    T indexRatio = fromMedium / toMedium;

    T delta = static_cast<T>(1.0) - indexRatio * indexRatio * (static_cast<T>(1.0) - cosTetha * cosTetha);
    if(delta > static_cast<T>(0.0))
    {
        direction = indexRatio * direction + normal * (cosTetha * indexRatio - sqrt(delta));
        return true;
    }
    return false;
}

template<class T>
__device__ __host__
inline Ray<T> Ray<T>::RandomRayCosine(T xi0, T xi1,
                                      const Vector<3, T>& normal,
                                      const Vector<3, T>& position)
{
    Vector<3, T> randomDir;
    randomDir[0] = sqrt(xi0) * cos(static_cast<T>(2.0) * MathConstants::Pi * xi1);
    randomDir[1] = sqrt(xi0) * sin(static_cast<T>(2.0) * MathConstants::Pi * xi1);
    randomDir[2] = sqrt(static_cast<T>(1.0) - xi0);

    Quaternion<T> q = Quaternion<T>::RotationBetweenZAxis(normal);
    Vector<3, T> rotatedDir = q.ApplyRotation(randomDir);
    return Ray(rotatedDir, position);
}

template<class T>
__device__ __host__
inline Ray<T> Ray<T>::RandomRayUnfirom(T xi0, T xi1,
                                       const Vector<3, T>& normal,
                                       const Vector<3, T>& position)
{
    Vector<3, T> randomDir;
    randomDir[0] = sqrt(static_cast<T>(1.0) - xi0 * xi0) * cos(static_cast<T>(2.0) * MathConstants::Pi * xi1);
    randomDir[1] = sqrt(static_cast<T>(1.0) - xi0 * xi0) * sin(static_cast<T>(2.0) * MathConstants::Pi * xi1);
    randomDir[2] = xi0;

    Quaternion<T> q = Quaternion<T>::RotationBetweenZAxis(normal);
    randomDir = q.ApplyRotation(randomDir);
    return Ray(randomDir, position);
}

template<class T>
__device__ __host__
inline Ray<T> Ray<T>::NormalizeDir() const
{
    return Ray(direction.Normalize(), position);
}

template<class T>
__device__ __host__
inline Ray<T>& Ray<T>::NormalizeDirSelf()
{
    direction.NormalizeSelf();
    return *this;
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
    return Ray<T>((mat * Vector<4, T>(direction, static_cast<T>(0.0))).Normalize(),
                  mat * Vector<4, T>(position, static_cast<T>(1.0)));
}

template<class T>
__device__ __host__
inline Ray<T>& Ray<T>::TransformSelf(const Matrix<4, T>& mat)
{
    direction = (mat * Vector<4, T>(direction, static_cast<T>(0.0))).Normalize();
    position = mat * Vector<4, T>(position, static_cast<T>(1.0));
    return *this;
}

template<class T>
__device__ __host__
inline Vector<3, T> Ray<T>::AdvancedPos(T t) const
{
    return position + t * direction;
}
