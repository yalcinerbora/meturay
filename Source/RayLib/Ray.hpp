#pragma once

template<class T>
__device__ __host__ HYBRID_INLINE
constexpr Ray<T>::Ray(const Vector<3, T>& direction, const Vector<3, T>& position)
    : direction(direction)
    , position(position)
{}

template<class T>
__device__ __host__ HYBRID_INLINE
constexpr Ray<T>::Ray(const Vector<3, T> vec[2])
    : direction(vec[0])
    , position(vec[1])
{}

template<class T>
__device__ __host__ HYBRID_INLINE
Ray<T>& Ray<T>::operator=(const Vector<3, T> vec[2])
{
    direction = vec[0];
    position = vec[1];
    return *this;
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
const Vector<3, T>& Ray<T>::getDirection() const
{
    return direction;
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
const Vector<3, T>& Ray<T>::getPosition() const
{
    return position;
}

template<class T>
__device__ __host__ HYBRID_INLINE
bool Ray<T>::IntersectsSphere(Vector<3, T>& intersectPos, T& t,
                              const Vector<3, T>& sphereCenter,
                              T sphereRadius) const
{
    // Clang min definition is only on std namespace
    // this is a crappy workaround
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    // RayTracing Gems
    // Chapter 7: Precision Improvements for Ray/Sphere Intersection
    // This is similar to the geometric solution below
    // with two differences
    //
    // 1st: Beam half length square (discriminant variable)
    // is calculated using two vector differences instead of
    // Pythagorean theorem (r^2 - beamNormalLength^2).
    // METUray almost always holds the direction vector
    // normalized unless the ray is under inverse transformation
    // of an object's local transformation
    // (and that transformation has  a scale).
    // So we need to normalize the direction vector to properly
    // find out the discriminant
    //
    // 2nd: closer t value is calculated classically; however
    // further t value is calculated differently which I could
    // not explain geometrically probably a reduction of some
    // algebraic expression which improves numerical accuracy
    //
    // First improvement, has higher accuracy when
    // extremely large sphere is being intersected
    //
    // Second one is for spheres that is far away
    T dirLengthInv = 1.0f / direction.Length();
    Vector<3, T> dirNorm = direction * dirLengthInv;
    Vector<3, T> centerDir = sphereCenter - position;
    T beamCenterDist = dirNorm.Dot(centerDir);
    T cDirLengthSqr = centerDir.LengthSqr();

    // Below code is from the source
    Vector<3, T> remedyTerm = centerDir - beamCenterDist * dirNorm;
    T discriminant = sphereRadius * sphereRadius - remedyTerm.LengthSqr();
    if(discriminant >= 0)
    {
        T beamHalfLength = sqrt(discriminant);

        T t0 = (beamCenterDist >= 0)
                    ? (beamCenterDist + beamHalfLength)
                    : (beamCenterDist - beamHalfLength);
        T t1 = (cDirLengthSqr - sphereRadius * sphereRadius) / t0;

        // TODO: is there a better way to do this?
        // Select a T
        t = FLT_MAX;
        if(t0 > 0) t = min(t, t0);
        if(t1 > 0) t = min(t, t1);
        if(t != FLT_MAX)
        {
            t *= dirLengthInv;
            intersectPos = position + t * direction;
            return true;
        }
    }
    return false;

    // Geometric solution
    //Vector<3, T> centerDir = sphereCenter - position;
    //T beamCenterDistance = centerDir.Dot(direction);
    //T beamNormalLengthSqr = (centerDir.Dot(centerDir) -
    //                         beamCenterDistance * beamCenterDistance);
    //T beamHalfLengthSqr = sphereRadius * sphereRadius - beamNormalLengthSqr;
    //if(beamHalfLengthSqr > 0)
    //{
    //    // Inside Square
    //    T beamHalfLength = sqrt(beamHalfLengthSqr);
    //    T t0 = beamCenterDistance - beamHalfLength;
    //    T t1 = beamCenterDistance + beamHalfLength;

    //    t = (fabs(t0) <= fabs(t1)) ? t0 : t1;
    //    intersectPos = position + t * direction;
    //    return true;
    //}
    //return false;
}

template<class T>
__device__ __host__ HYBRID_INLINE
bool Ray<T>::IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
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
__device__ __host__ HYBRID_INLINE
bool Ray<T>::IntersectsTriangle(Vector<3, T>& baryCoords, T& t,
                                const Vector<3, T>& t0,
                                const Vector<3, T>& t1,
                                const Vector<3, T>& t2,
                                bool cullFace) const
{
    // Moller-Trumbore
    // Ray-Tri Intersection
    Vector<3, T> e0 = t1 - t0;
    Vector<3, T> e1 = t2 - t0;
    Vector<3, T> p = Cross<T>(direction, e1);
    T det = e0.Dot(p);

    if((cullFace && (det < MathConstants::SmallEpsilon)) ||
       // Ray-Tri nearly parallel skip
       (abs(det) < MathConstants::SmallEpsilon))
        return false;

    T invDet = 1 / det;

    Vector<3, T> tVec = position - t0;
    baryCoords[0] = tVec.Dot(p) * invDet;
    // Early Skip
    if(baryCoords[0] < 0 || baryCoords[0] > 1)
        return false;

    Vector<3, T> qVec = Cross<T>(tVec, e0);
    baryCoords[1] = direction.Dot(qVec) * invDet;
    // Early Skip 2
    if((baryCoords[1] < 0) || (baryCoords[1] + baryCoords[0]) > 1)
        return false;

    t = e1.Dot(qVec) * invDet;
    if(t <= MathConstants::SmallEpsilon)
        return false;

    // Calculate C
    baryCoords[2] = 1 - baryCoords[0] - baryCoords[1];
    baryCoords = Vector<3, T>(baryCoords[2],
                              baryCoords[0],
                              baryCoords[1]);
    return true;

    //// Matrix Solution
    //// Kramer's Rule
    //Vector<3, T> abDiff = t0 - t1;
    //Vector<3, T> acDiff = t0 - t2;
    //Vector<3, T> aoDiff = t0 - position;

    //if(cullFace)
    //{
    //    // TODO this is wrong??
    //    Vector<3, T> normal = Cross(abDiff, acDiff).Normalize();
    //    T cos = direction.Dot(normal);
    //    if(cos > 0) return false;
    //}

    //Vector<3, T> aData[] = {abDiff, acDiff, direction};
    //Vector<3, T> betaAData[] = {aoDiff, acDiff, direction};
    //Vector<3, T> gammaAData[] = {abDiff, aoDiff, direction};
    //Vector<3, T> tAData[] = {abDiff, acDiff, aoDiff};

    //Matrix<3, T> A = Matrix<3, T>(aData);
    //Matrix<3, T> betaA = Matrix<3, T>(betaAData);
    //Matrix<3, T> gammaA = Matrix<3, T>(gammaAData);
    //Matrix<3, T> tA = Matrix<3, T>(tAData);

    //T aDetInv = 1.0f / A.Determinant();
    //T beta = betaA.Determinant() * aDetInv;
    //T gamma = gammaA.Determinant() * aDetInv;
    //T alpha = 1.0f - beta - gamma;
    //T rayT = tA.Determinant() * aDetInv;

    //if(beta >= 0.0f && beta <= 1.0f &&
    //   gamma >= 0.0f && gamma <= 1.0f &&
    //   alpha >= 0.0f && alpha <= 1.0f &&
    //   rayT >= 0.0f)
    //{
    //    baryCoords = Vector<3, T>(alpha, beta, gamma);
    //    t = rayT;
    //    return true;
    //}
    //else return false;
}

template<class T>
__device__ __host__ HYBRID_INLINE
bool Ray<T>::IntersectsPlane(Vector<3, T>& intersectPos, T& t,
                             const Vector<3, T>& planePos,
                             const Vector<3, T>& normal)
{
    T nDotD = normal.Dot(direction);
    // Nearly parallel
    if(abs(nDotD) <= MathConstants::Epsilon)
    {
        t = INFINITY;
        return false;
    }
    t = (planePos - position).Dot(normal) / nDotD;
    intersectPos = position + t * direction;
    return true;
}

template<class T>
__device__ __host__ HYBRID_INLINE
bool Ray<T>::IntersectsAABB(const Vector<3, T>& aabbMin,
                            const Vector<3, T>& aabbMax,
                            const Vector<2, T>& tMinMax) const
{
    // CPU code max/min is on std namespace but CUDA has its global namespace
    #ifndef __CUDA_ARCH__
        using namespace std;
    #endif

    Vector<3, T> invD = Vector<3, T>(1) / direction;
    Vector<3, T> t0 = (aabbMin - position) * invD;
    Vector<3, T> t1 = (aabbMax - position) * invD;

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
__device__ __host__ HYBRID_INLINE
bool Ray<T>::IntersectsAABB(Vector<3, T>& pos, T& tOut,
                            const Vector<3, T>& aabbMin,
                            const Vector<3, T>& aabbMax,
                            const Vector<2, T>& tMinMax) const
{
    Vector<3, T> invD = Vector<3, T>(1) / direction;
    Vector<3, T> t0 = (aabbMin - position) * invD;
    Vector<3, T> t1 = (aabbMax - position) * invD;

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
__device__ __host__ HYBRID_INLINE [[nodiscard]]
Ray<T> Ray<T>::Reflect(const Vector<3, T>& normal) const
{
    Vector<3, T> nDir = direction;
    nDir = static_cast<T>(2.0) * nDir.Dot(normal) * normal - nDir;
    return Ray(nDir, position);
}

template<class T>
__device__ __host__ HYBRID_INLINE
Ray<T>& Ray<T>::ReflectSelf(const Vector<3, T>& normal)
{
    Vector<3, T> nDir = direction;
    direction = (static_cast<T>(2.0) * nDir.Dot(normal) * normal - nDir);
    return *this;
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
bool Ray<T>::Refract(Ray& out, const Vector<3, T>& normal,
                     T fromMedium, T toMedium) const
{
    // Convention of wi (ray.direction) and normal is as follows
    //          wo (out.direction)
    //          ^
    //         /
    //        /    toMedium (IOR)
    //--------------- Boundary
    //     /  |    fromMedium (IOR)
    //    /   |
    //   /    |
    //  v     v
    //  wi  normal

    constexpr T Zero = static_cast<T>(0.0);
    constexpr T One = static_cast<T>(1.0);

    T indexRatio = fromMedium / toMedium;
    T cosIn = normal.Dot(direction);
    T sinInSqr = max(Zero, One - cosIn * cosIn);
    T sinOutSqr = indexRatio * indexRatio * sinInSqr;
    T cosOut = sqrt(max(Zero, One - sinOutSqr));

    // Check total internal reflection
    if(sinOutSqr >= One) return false;

    out.direction = (indexRatio * (-direction) +
                     (indexRatio * cosIn - cosOut) * normal);
    out.position = position;
    return true;
}

template<class T>
__device__ __host__ HYBRID_INLINE
bool Ray<T>::RefractSelf(const Vector<3, T>& normal,
                                T fromMedium, T toMedium)
{
    Ray<T> outRay;
    bool result = Refract(outRay, normal, fromMedium, toMedium);
    direction = outRay.direction;
    return result;
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
Ray<T> Ray<T>::RandomRayCosine(T xi0, T xi1,
                               const Vector<3, T>& normal,
                               const Vector<3, T>& position)
{
    Vector<3, T> randomDir;
    randomDir[0] = sqrt(xi0) * cos(static_cast<T>(2.0) * MathConstants::Pi * xi1);
    randomDir[1] = sqrt(xi0) * sin(static_cast<T>(2.0) * MathConstants::Pi * xi1);
    randomDir[2] = sqrt(static_cast<T>(1.0) - xi0);

    Quaternion<T> q = Quat::RotationBetweenZAxis(normal);
    Vector<3, T> rotatedDir = q.ApplyRotation(randomDir);
    return Ray(rotatedDir, position);
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
Ray<T> Ray<T>::RandomRayUnfirom(T xi0, T xi1,
                                const Vector<3, T>& normal,
                                const Vector<3, T>& position)
{
    Vector<3, T> randomDir;
    randomDir[0] = sqrt(static_cast<T>(1.0) - xi0 * xi0) * cos(static_cast<T>(2.0) * MathConstants::Pi * xi1);
    randomDir[1] = sqrt(static_cast<T>(1.0) - xi0 * xi0) * sin(static_cast<T>(2.0) * MathConstants::Pi * xi1);
    randomDir[2] = xi0;

    Quaternion<T> q = Quat::RotationBetweenZAxis(normal);
    randomDir = q.ApplyRotation(randomDir);
    return Ray(randomDir, position);
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
Ray<T> Ray<T>::NormalizeDir() const
{
    return Ray(direction.Normalize(), position);
}

template<class T>
__device__ __host__ HYBRID_INLINE
Ray<T>& Ray<T>::NormalizeDirSelf()
{
    direction.NormalizeSelf();
    return *this;
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
Ray<T> Ray<T>::Advance(T t) const
{
    return Ray(direction, position + t * direction);
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
Ray<T> Ray<T>::Advance(T t, const Vector<3, T>& dir) const
{
    return Ray(direction, position + t * dir);
}

template<class T>
__device__ __host__ HYBRID_INLINE
Ray<T>& Ray<T>::AdvanceSelf(T t)
{
    position += t * direction;
    return *this;
}

template<class T>
__device__ __host__ HYBRID_INLINE
Ray<T>& Ray<T>::AdvanceSelf(T t, const Vector<3, T>& dir)
{
    position += t * dir;
    return *this;
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
Ray<T> Ray<T>::Transform(const Quaternion<T>& q) const
{
    return Ray<T>(q.ApplyRotation(direction),
                  q.ApplyRotation(position));
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
Ray<T> Ray<T>::Transform(const Matrix<3, T>& mat) const
{
    return Ray<T>(mat * direction,
                  mat * position);
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
Ray<T> Ray<T>::Transform(const Matrix<4, T>& mat) const
{
    return Ray<T>(mat * Vector<4, T>(direction, static_cast<T>(0.0)),
                  mat * Vector<4, T>(position, static_cast<T>(1.0)));
}

template<class T>
__device__ __host__ HYBRID_INLINE
Ray<T> Ray<T>::TransformSelf(const Quaternion<T>& q)
{
    Ray<T> r = Transform(q);
    (*this) = r;
    return *this;
}

template<class T>
__device__ __host__ HYBRID_INLINE
Ray<T>& Ray<T>::TransformSelf(const Matrix<3, T>& mat)
{
    Ray<T> r = Transform(mat);
    (*this) = r;
    return *this;
}

template<class T>
__device__ __host__ HYBRID_INLINE
Ray<T>& Ray<T>::TransformSelf(const Matrix<4, T>& mat)
{
    Ray<T> r = Transform(mat);
    (*this) = r;
    return *this;
}

template<class T>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
Vector<3, T> Ray<T>::AdvancedPos(T t) const
{
    return position + t * direction;
}

template<>
__device__ __host__ HYBRID_INLINE [[nodiscard]]
Ray<float> Ray<float>::Nudge(const Vector3f& dir, float curvatureOffset) const
{
    // From RayTracing Gems I
    // Chapter 6
    static constexpr float ORIGIN = 1.0f / 32.0f;
    static constexpr float FLOAT_SCALE = 1.0f / 65536.0f;
    static constexpr float INT_SCALE = 256.0f;

    const Vector3f& p = position;
    Vector3i ofi = Vector3i(INT_SCALE * dir[0],
                            INT_SCALE * dir[1],
                            INT_SCALE * dir[2]);

    Vector3f pointI;
    #ifdef __CUDA_ARCH__
        pointI[0] = __int_as_float(__float_as_int(p[0]) + ((p[0] < 0) ? -ofi[0] : ofi[0]));
        pointI[1] = __int_as_float(__float_as_int(p[1]) + ((p[1] < 0) ? -ofi[1] : ofi[1]));
        pointI[2] = __int_as_float(__float_as_int(p[2]) + ((p[2] < 0) ? -ofi[2] : ofi[2]));
    #else
        // CPU code (this will be optimized out)
        // and it is not UB
        static_assert(sizeof(int32_t) == sizeof(float));
        Vector3i pInt;
        memcpy(&(pInt[0]), &(p[0]), sizeof(float));
        memcpy(&(pInt[1]), &(p[1]), sizeof(float));
        memcpy(&(pInt[2]), &(p[2]), sizeof(float));

        pInt[0] += ((p[0] < 0) ? -ofi[0] : ofi[0]);
        pInt[1] += ((p[1] < 0) ? -ofi[1] : ofi[1]);
        pInt[2] += ((p[2] < 0) ? -ofi[2] : ofi[2]);

        memcpy(&(pointI[0]), &(pInt[0]), sizeof(float));
        memcpy(&(pointI[1]), &(pInt[1]), sizeof(float));
        memcpy(&(pointI[2]), &(pInt[2]), sizeof(float));
    #endif

    // Find the next floating point towards
    // Either use an epsilon (float_scale in this case)
    // or use the calculated offset
    Vector3f nextPos;
    nextPos[0] = (fabs(p[0]) < ORIGIN) ? (p[0] + FLOAT_SCALE * dir[0]) : pointI[0];
    nextPos[1] = (fabs(p[1]) < ORIGIN) ? (p[1] + FLOAT_SCALE * dir[1]) : pointI[1];
    nextPos[2] = (fabs(p[2]) < ORIGIN) ? (p[2] + FLOAT_SCALE * dir[2]) : pointI[2];

    if(curvatureOffset != 0.0f)
    {
        nextPos[0] = nextPos[0] + dir[0] * curvatureOffset;
        nextPos[1] = nextPos[1] + dir[1] * curvatureOffset;
        nextPos[2] = nextPos[2] + dir[2] * curvatureOffset;
    }

    return Ray(direction, nextPos);
}

template<>
__device__ __host__ HYBRID_INLINE
Ray<double> Ray<double>::Nudge(const Vector3d& dir, double curvatureOffset) const
{
    // From RayTracing Gems I
    // Chapter 6
    static constexpr double ORIGIN = 1.0 / 32.0;
    static constexpr double FLOAT_SCALE = 1.0 / 65536.0;
    static constexpr double INT_SCALE = 256.0;

    const Vector3d& p = position;

    Vector<3, int64_t> ofi(INT_SCALE * dir[0],
                           INT_SCALE * dir[1],
                           INT_SCALE * dir[2]);

    Vector3d pointI;
    // Utilize memcpy here, it is not UBO
    // and should be optimized since it is in register space
    static_assert(sizeof(int64_t) == sizeof(double));
    Vector<3, int64_t> pInt;
    memcpy(&(pInt[0]), &(p[0]), sizeof(double));
    memcpy(&(pInt[1]), &(p[1]), sizeof(double));
    memcpy(&(pInt[2]), &(p[2]), sizeof(double));

    pInt[0] += ((p[0] < 0) ? -ofi[0] : ofi[0]);
    pInt[1] += ((p[1] < 0) ? -ofi[1] : ofi[1]);
    pInt[2] += ((p[2] < 0) ? -ofi[2] : ofi[2]);

    memcpy(&(pointI[0]), &(pInt[0]), sizeof(double));
    memcpy(&(pointI[1]), &(pInt[1]), sizeof(double));
    memcpy(&(pointI[2]), &(pInt[2]), sizeof(double));

    // Find the next floating point towards
    // Either use an epsilon (float_scale in this case)
    // or use the calculated offset
    Vector3d nextPos;
    nextPos[0] = (fabs(p[0]) < ORIGIN) ? (p[0] + FLOAT_SCALE * dir[0]) : pointI[0];
    nextPos[1] = (fabs(p[1]) < ORIGIN) ? (p[1] + FLOAT_SCALE * dir[1]) : pointI[1];
    nextPos[2] = (fabs(p[2]) < ORIGIN) ? (p[2] + FLOAT_SCALE * dir[2]) : pointI[2];

    if(curvatureOffset != 0.0)
    {
        nextPos[0] = nextPos[0] + dir[0] * curvatureOffset;
        nextPos[1] = nextPos[1] + dir[1] * curvatureOffset;
        nextPos[2] = nextPos[2] + dir[2] * curvatureOffset;
    }
    return Ray(direction, nextPos);
}

template<class T>
__device__ __host__ HYBRID_INLINE
Ray<T>& Ray<T>::NudgeSelf(const Vector<3, T>& dir, T curvatureOffset)
{
    Ray<T> r = Nudge(dir, curvatureOffset);
    (*this) = r;
    return *this;
}