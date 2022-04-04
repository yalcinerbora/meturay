template <int N, class T>
template <class C, typename>
__device__ __host__ HYBRID_INLINE
Matrix<N, T>::Matrix(C t)
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] = static_cast<C>(t);
    }
}

template <int N, class T>
template <class C, typename>
__device__ __host__ HYBRID_INLINE
Matrix<N, T>::Matrix(const C* data)
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] = static_cast<C>(data[i]);
    }
}

template <int N, class T>
template <class... Args, typename>
__device__ __host__ HYBRID_INLINE
constexpr Matrix<N, T>::Matrix(const Args... dataList)
    : matrix{static_cast<T>(dataList) ...}
{
    static_assert(sizeof...(dataList) == N * N, "Matrix constructor should have exact "
                  "same count of template count "
                  "as arguments");
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T>::Matrix(const Vector<N, T> columns[])
{
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        const Vector<N, T>& vec = columns[i];
        UNROLL_LOOP
        for(int j = 0; j < N; j++)
        {
            matrix[i * N + j] = vec[j];
        }
    }
}

template <int N, class T>
template <int M, typename>
__device__ __host__ HYBRID_INLINE
Matrix<N, T>::Matrix(const Matrix<M, T>& other)
{
    static_assert(M >= N, "enable_if sanity check.");
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        UNROLL_LOOP
        for(int j = 0; j < N; j++)
        {
            if(i < M && j < M)
                matrix[i * N + j] = other[i * M + j];
            else if(i == N && j == N)
                matrix[i * M + j] = 1;
            else
                matrix[i * M + j] = 0;
        }
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T>::operator T* ()
{
    return matrix;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T>::operator const T* () const
{
    return matrix;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
T& Matrix<N, T>::operator[](int i)
{
    return matrix[i];
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
const T& Matrix<N, T>::operator[](int i) const
{
    return matrix[i];
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
T& Matrix<N, T>::operator()(int row, int column)
{
    return matrix[column * N + row];
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
const T& Matrix<N, T>::operator()(int row, int column) const
{
    return matrix[column * N + row];
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Matrix<N, T>::operator+=(const Matrix& right)
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] += right.matrix[i];
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Matrix<N, T>::operator-=(const Matrix& right)
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] -= right.matrix[i];
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Matrix<N, T>::operator*=(const Matrix& right)
{
    Matrix m = (*this) * right;
    *this = m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Matrix<N, T>::operator*=(T right)
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] *= right;
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Matrix<N, T>::operator/=(const Matrix& right)
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] /= right.matrix[i];
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
void Matrix<N, T>::operator/=(T right)
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] /= right;
    }
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::operator+(const Matrix& right) const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] + right.matrix[i];
    }
    return m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::operator-(const Matrix& right) const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] - right.matrix[i];
    }
    return m;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
SignedEnable<Q, Matrix<N, T>> Matrix<N, T>::operator-() const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m.matrix[i] = -matrix[i];
    }
    return m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::operator/(const Matrix& right) const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] / right.matrix[i];
    }
    return m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::operator/(T right) const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] / right;
    }
    return m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::operator*(const Matrix& right) const
{
    Matrix m;
    UNROLL_LOOP
    for(int j = 0; j < N; j++)
    {
        UNROLL_LOOP
        for(int i = 0; i < N; i++)
        {
            T result = 0;
            UNROLL_LOOP
            for(int k = 0; k < N; k++)
            {
                result += matrix[j + N * k] * right[i * N + k];
            }
            // Dot Product
            m(j, i) = result;
        }
    }
    return m;
}

template <int N, class T>
template <int M>
__device__ __host__ HYBRID_INLINE
Vector<M, T> Matrix<N, T>::operator*(const Vector<M, T>& right) const
{
    static_assert(M <= N, "Cannot Multiply with large vector.");

    Vector<M, T> v;
    UNROLL_LOOP
    for(int i = 0; i < M; i++)
    {
        T result = 0;
        UNROLL_LOOP
        for(int k = 0; k < M; k++)
        {
            result += matrix[i + N * k] * right[k];
        }
        // Dot Product
        v[i] = result;
    }
    return v;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::operator*(T right) const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m.matrix[i] = matrix[i] * right;
    }
    return m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
bool Matrix<N, T>::operator==(const Matrix& right) const
{
    bool eq = true;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        eq &= matrix[i] == right.matrix[i];
    }
    return eq;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
bool Matrix<N, T>::operator!=(const Matrix& right) const
{
    return !(*this == right);
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
T Matrix<N, T>::Determinant() const
{
    if constexpr(N == 2)
        return Determinant2<T>(static_cast<const T*>(*this));
    else if constexpr(N == 3)
        return Determinant3<T>(static_cast<const T*>(*this));
    else
        return Determinant4<T>(static_cast<const T*>(*this));
    UNREACHABLE();
}

template <int N, class T>
template <class Q>
__device__ __host__
inline FloatEnable<Q, Matrix<N, T>> Matrix<N, T>::Inverse() const
{
    if constexpr(N == 2)
        return Inverse2<T>(static_cast<const T*>(*this));
    else if constexpr(N == 3)
        return Inverse3<T>(static_cast<const T*>(*this));
    else
        return Inverse4<T>(static_cast<const T*>(*this));
    UNREACHABLE();
}

template <int N, class T>
template <class Q>
__device__ __host__
inline FloatEnable<Q, Matrix<N, T>&> Matrix<N, T>::InverseSelf()
{
    if constexpr(N == 2)
        (*this) = Inverse2<T>(static_cast<const T*>(*this));
    else if constexpr(N == 3)
        (*this) = Inverse3<T>(static_cast<const T*>(*this));
    else
        (*this) = Inverse4<T>(static_cast<const T*>(*this));
    return *this;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::Transpose() const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N; i++)
    {
        UNROLL_LOOP
        for(int j = 0; j < N; j++)
        {
            m(j, i) = (*this)(i, j);
        }
    }
    return m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T>& Matrix<N, T>::TransposeSelf()
{
    UNROLL_LOOP
    for(int i = 1; i < N; i++)
    {
        UNROLL_LOOP
        for(int j = 0; j < i; j++)
        {
            // CARPRAZ SWAP
            T a = (*this)(i, j);
            (*this)(i, j) = (*this)(j, i);
            (*this)(j, i) = a;
        }
    }
    return *this;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::Clamp(const Matrix& minVal, const Matrix& maxVal) const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m[i] = min(max(minVal[i], matrix[i]), maxVal[i]);
    }
    return m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::Clamp(T minVal, T maxVal) const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m[i] = min(max(minVal, matrix[i]), maxVal);
    }
    return m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T>& Matrix<N, T>::ClampSelf(const Matrix& minVal, const Matrix& maxVal)
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] = min(max(minVal[i], matrix[i]), maxVal[i]);
    }
    return *this;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T>& Matrix<N, T>::ClampSelf(T minVal, T maxVal)
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] = min(max(minVal, matrix[i]), maxVal);
    }
    return *this;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
SignedEnable<Q, Matrix<N, T>> Matrix<N, T>::Abs() const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m[i] = abs(matrix[i]);
    }
    return m;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
SignedEnable<Q, Matrix<N, T>&> Matrix<N, T>::AbsSelf()
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] = abs(matrix[i]);
    }
    return *this;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Matrix<N, T>> Matrix<N, T>::Round() const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m[i] = round(matrix[i]);
    }
    return m;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Matrix<N, T>&> Matrix<N, T>::RoundSelf()
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] = round(matrix[i]);
    }
    return *this;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Matrix<N, T>> Matrix<N, T>::Floor() const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m[i] = floor(matrix[i]);
    }
    return m;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Matrix<N, T>&> Matrix<N, T>::FloorSelf()
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] = floor(matrix[i]);
    }
    return *this;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Matrix<N, T>> Matrix<N, T>::Ceil() const
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m[i] = ceil(matrix[i]);
    }
    return m;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Matrix<N, T>&> Matrix<N, T>::CeilSelf()
{
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        matrix[i] = ceil(matrix[i]);
    }
    return *this;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::Min(const Matrix& mat0, const Matrix& mat1)
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m[i] = min(mat0[i], mat1[i]);
    }
    return m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::Min(const Matrix& mat0, T t)
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m[i] = min(mat0[i], t);
    }
    return m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::Max(const Matrix& mat0, const Matrix& mat1)
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m[i] = max(mat0[i], mat1[i]);
    }
    return m;
}

template <int N, class T>
__device__ __host__ HYBRID_INLINE
Matrix<N, T> Matrix<N, T>::Max(const Matrix& mat0, T t)
{
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m[i] = max(mat0[i], t);
    }
    return m;
}

template <int N, class T>
template <class Q>
__device__ __host__ HYBRID_INLINE
FloatEnable<Q, Matrix<N, T>> Matrix<N, T>::Lerp(const Matrix& mat0, const Matrix& mat1, T t)
{
    assert(t >= 0 && t <= 1);
    Matrix m;
    UNROLL_LOOP
    for(int i = 0; i < N * N; i++)
    {
        m[i] = (1 - t) * mat0[i] + t * mat1[i];
    }
    return m;
}

template<class T>
__device__ __host__
T Determinant2(const T* m)
{
    return m[0] * m[3] - m[2] * m[1];
}

template<class T>
__device__ __host__
T Determinant3(const T* m)
{
    T det1 = m[0] * (m[4] * m[8] - m[7] * m[5]);
    T det2 = m[3] * (m[1] * m[8] - m[7] * m[2]);
    T det3 = m[6] * (m[1] * m[5] - m[4] * m[2]);
    return det1 - det2 + det3;
}

template<class T>
__device__ __host__
T Determinant4(const T* m)
{
    // Hard-coded should be most optimizer friendly
    // TODO: Maybe register size etc.. for GPU
    // YOLO
    T det1 = m[0] * (m[5] * m[10] * m[15]
                     + m[9] * m[14] * m[7]
                     + m[6] * m[11] * m[13]
                     - m[13] * m[10] * m[7]
                     - m[9] * m[6] * m[15]
                     - m[5] * m[14] * m[11]);

    T det2 = m[4] * (m[1] * m[10] * m[15]
                     + m[9] * m[14] * m[3]
                     + m[2] * m[11] * m[13]
                     - m[3] * m[10] * m[13]
                     - m[2] * m[9] * m[15]
                     - m[1] * m[11] * m[14]);

    T det3 = m[8] * (m[1] * m[6] * m[15]
                     + m[5] * m[14] * m[3]
                     + m[2] * m[7] * m[13]
                     - m[3] * m[6] * m[13]
                     - m[2] * m[5] * m[15]
                     - m[14] * m[7] * m[1]);

    T det4 = m[12] * (m[1] * m[6] * m[11]
                      + m[5] * m[10] * m[3]
                      + m[2] * m[7] * m[9]
                      - m[9] * m[6] * m[3]
                      - m[2] * m[5] * m[11]
                      - m[1] * m[10] * m[7]);
    return det1 - det2 + det3 - det4;
}

template<class T>
__device__ __host__
Matrix<2, T> Inverse2(const T* m)
{
    Matrix<2, T> result;
    T detRecip = 1 / Determinant2<T>(m);

    result(0, 0) = detRecip * m[3];
    result(0, 1) = detRecip * m[1] * -1;
    result(1, 0) = detRecip * m[2] * -1;
    result(1, 1) = detRecip * m[0];
    return result;
}

template<class T>
__device__ __host__
Matrix<3, T> Inverse3(const T* m)
{
    T m11 = m[4] * m[8] - m[7] * m[5];
    T m12 = -(m[1] * m[8] - m[7] * m[2]);
    T m13 = m[1] * m[5] - m[4] * m[2];

    T m21 = -(m[3] * m[8] - m[6] * m[5]);
    T m22 = m[0] * m[8] - m[6] * m[2];
    T m23 = -(m[0] * m[5] - m[3] * m[2]);

    T m31 = m[3] * m[7] - m[6] * m[4];
    T m32 = -(m[0] * m[7] - m[6] * m[1]);
    T m33 = m[0] * m[4] - m[3] * m[1];

    T det = m[0] * m11 + m[3] * m12 + m[6] * m13;
    T detInv = 1 / det;
    return detInv * Matrix<3, T>(m11, m12, m13,
                                 m21, m22, m23,
                                 m31, m32, m33);
}

template<class T>
__device__ __host__
Matrix<4, T> Inverse4(const T* m)
{
    // MESA GLUT Copy Paste
    Matrix<4, T> inv;

    inv[0] = m[5] * m[10] * m[15] -
        m[5] * m[11] * m[14] -
        m[9] * m[6] * m[15] +
        m[9] * m[7] * m[14] +
        m[13] * m[6] * m[11] -
        m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] +
        m[4] * m[11] * m[14] +
        m[8] * m[6] * m[15] -
        m[8] * m[7] * m[14] -
        m[12] * m[6] * m[11] +
        m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] -
        m[4] * m[11] * m[13] -
        m[8] * m[5] * m[15] +
        m[8] * m[7] * m[13] +
        m[12] * m[5] * m[11] -
        m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] +
        m[4] * m[10] * m[13] +
        m[8] * m[5] * m[14] -
        m[8] * m[6] * m[13] -
        m[12] * m[5] * m[10] +
        m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] +
        m[1] * m[11] * m[14] +
        m[9] * m[2] * m[15] -
        m[9] * m[3] * m[14] -
        m[13] * m[2] * m[11] +
        m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] -
        m[0] * m[11] * m[14] -
        m[8] * m[2] * m[15] +
        m[8] * m[3] * m[14] +
        m[12] * m[2] * m[11] -
        m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] +
        m[0] * m[11] * m[13] +
        m[8] * m[1] * m[15] -
        m[8] * m[3] * m[13] -
        m[12] * m[1] * m[11] +
        m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] -
        m[0] * m[10] * m[13] -
        m[8] * m[1] * m[14] +
        m[8] * m[2] * m[13] +
        m[12] * m[1] * m[10] -
        m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] -
        m[1] * m[7] * m[14] -
        m[5] * m[2] * m[15] +
        m[5] * m[3] * m[14] +
        m[13] * m[2] * m[7] -
        m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] +
        m[0] * m[7] * m[14] +
        m[4] * m[2] * m[15] -
        m[4] * m[3] * m[14] -
        m[12] * m[2] * m[7] +
        m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] -
        m[0] * m[7] * m[13] -
        m[4] * m[1] * m[15] +
        m[4] * m[3] * m[13] +
        m[12] * m[1] * m[7] -
        m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] +
        m[0] * m[6] * m[13] +
        m[4] * m[1] * m[14] -
        m[4] * m[2] * m[13] -
        m[12] * m[1] * m[6] +
        m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
        m[1] * m[7] * m[10] +
        m[5] * m[2] * m[11] -
        m[5] * m[3] * m[10] -
        m[9] * m[2] * m[7] +
        m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
        m[0] * m[7] * m[10] -
        m[4] * m[2] * m[11] +
        m[4] * m[3] * m[10] +
        m[8] * m[2] * m[7] -
        m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
        m[0] * m[7] * m[9] +
        m[4] * m[1] * m[11] -
        m[4] * m[3] * m[9] -
        m[8] * m[1] * m[7] +
        m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
        m[0] * m[6] * m[9] -
        m[4] * m[1] * m[10] +
        m[4] * m[2] * m[9] +
        m[8] * m[1] * m[6] -
        m[8] * m[2] * m[5];

    T det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    det = 1 / det;

    return inv * det;
}

template<int N, class T>
__device__ __host__
Matrix<N, T> operator*(float t, const Matrix<N, T>& mat)
{
    return mat * t;
}

// Spacial Matrix4x4 -> Matrix3x3
template<class T>
static __device__ __host__
Matrix<4, T> ToMatrix4x4(const Matrix<3, T>& m)
{
    return Matrix<4, T>(m[0], m[3], m[6], 0,
                        m[1], m[4], m[7], 0,
                        m[2], m[5], m[8], 0,
                        0,    0,    0,    1);
}

template<class T, typename>
__device__ __host__
Vector<3, T> TransformGen::ExtractScale(const Matrix<4, T>& m)
{
    // This is not proper!
    // This should fail if transform matrix has shear
    // (didn't tested tho)
    T sX = Vector<3, T>(m[0], m[4], m[8]).Length();
    T sY = Vector<3, T>(m[1], m[5], m[9]).Length();
    T sZ = Vector<3, T>(m[2], m[6], m[10]).Length();
    return Vector<3, T>(sX, sY, sZ);
}

template<class T, typename>
__device__ __host__
Vector<3, T> TransformGen::ExtractTranslation(const Matrix<4, T>& m)
{
    return Vector<3, T>(m[12], m[13], m[14]);
}

template<class T, typename>
__device__ __host__
Matrix<4, T> TransformGen::Translate(const Vector<3, T>& v)
{
    //  1       0       0       tx
    //  0       1       0       ty
    //  0       0       1       tz
    //  0       0       0       1
    return Matrix<4, T>(1,    0,    0,    0,
                        0,    1,    0,    0,
                        0,    0,    1,    0,
                        v[0], v[1], v[2], 1);
}

template<class T, typename>
__device__ __host__
Matrix<4, T> TransformGen::Scale(T s)
{
    //  s       0       0       0
    //  0       s       0       0
    //  0       0       s       0
    //  0       0       0       1
    return Matrix<4, T>(s, 0, 0, 0,
                        0, s, 0, 0,
                        0, 0, s, 0,
                        0, 0, 0, 1);
}

template<class T, typename>
__device__ __host__
Matrix<4, T> TransformGen::Scale(T x, T y, T z)
{
    //  sx      0       0       0
    //  0       sy      0       0
    //  0       0       sz      0
    //  0       0       0       1
    return Matrix<4, T>(x, 0, 0, 0,
                        0, y, 0, 0,
                        0, 0, z, 0,
                        0, 0, 0, 1);
}

template<class T, typename>
__device__ __host__
Matrix<4, T> TransformGen::Rotate(T angle, const Vector<3, T>& axis)
{
    //  r       r       r       0
    //  r       r       r       0
    //  r       r       r       0
    //  0       0       0       1
    T tmp1, tmp2;

    T cosAngle = cos(angle);
    T sinAngle = sin(angle);
    T t = 1 - cosAngle;

    tmp1 = axis[0] * axis[1] * t;
    tmp2 = axis[2] * sinAngle;
    T m21 = tmp1 + tmp2;
    T m12 = tmp1 - tmp2;

    tmp1 = axis[0] * axis[2] * t;
    tmp2 = axis[1] * sinAngle;
    T m31 = tmp1 - tmp2;
    T m13 = tmp1 + tmp2;

    tmp1 = axis[1] * axis[2] * t;
    tmp2 = axis[0] * sinAngle;
    T m32 = tmp1 + tmp2;
    T m23 = tmp1 - tmp2;

    T m11 = cosAngle + axis[0] * axis[0] * t;
    T m22 = cosAngle + axis[1] * axis[1] * t;
    T m33 = cosAngle + axis[2] * axis[2] * t;

    return Matrix<4, T>(m11, m21, m31, 0,
                        m12, m22, m32, 0,
                        m13, m23, m33, 0,
                        0,   0,   0,   1);
}

template<class T, typename>
__device__ __host__
Matrix<4, T> TransformGen::Rotate(const Quaternion<T>& q)
{
    //QuatF q = quat.Normalize();
    Matrix<4, T> result;
    T xx = q[1] * q[1];
    T xy = q[1] * q[2];
    T xz = q[1] * q[3];
    T xw = q[1] * q[0];
    T yy = q[2] * q[2];
    T yz = q[2] * q[3];
    T yw = q[2] * q[0];
    T zz = q[3] * q[3];
    T zw = q[3] * q[0];
    result[0] = 1 - (2 * (yy + zz));
    result[4] = (2 * (xy - zw));
    result[8] = (2 * (xz + yw));
    result[12] = 0;

    result[1] = (2 * (xy + zw));
    result[5] = 1 - (2 * (xx + zz));
    result[9] = (2 * (yz - xw));
    result[13] = 0;

    result[2] = (2 * (xz - yw));
    result[6] = (2 * (yz + xw));
    result[10] = 1 - (2 * (xx + yy));
    result[14] = 0;

    result[3] = 0;
    result[7] = 0;
    result[11] = 0;
    result[15] = 1;
    return result;
}

template<class T, typename>
__device__ __host__
Matrix<4, T> TransformGen::Perspective(T fovXRadians, T aspectRatio,
                                       T nearPlane, T farPlane)
{
    //  p       0       0       0
    //  0       p       0       0
    //  0       0       p       -1
    //  0       0       p       0
    T f = 1 / tan(fovXRadians * static_cast<T>(0.5));
    T m33 = (farPlane + nearPlane) / (nearPlane - farPlane);
    T m34 = (2 * farPlane * nearPlane) / (nearPlane - farPlane);
    //float m33 = farPlane / (nearPlane - farPlane);
    //float m34 = (nearPlane * farPlane) / (nearPlane - farPlane);

    return Matrix<4, T>(f, 0, 0, 0,
                        0, f * aspectRatio, 0, 0,
                        0, 0, m33, -1,
                        0, 0, m34, 0);
}

template<class T, typename>
__device__ __host__
Matrix<4, T> TransformGen::Ortogonal(T left, T right,
                                     T top, T bottom,
                                     T nearPlane, T farPlane)
{
    //  orto    0       0       0
    //  0       orto    0       0
    //  0       0       orto    0
    //  orto    orto    orto    1
    T xt = -((right + left) / (right - left));
    T yt = -((top + bottom) / (top - bottom));
    T zt = -((farPlane + nearPlane) / (farPlane - nearPlane));
    T xs = 2 / (right - left);
    T ys = 2 / (top - bottom);
    T zs = 2 / (farPlane - nearPlane);
    return  Matrix<4, T>(xs,  0,  0, 0,
                          0, ys,  0, 0,
                          0,  0, zs, 0,
                         xt, yt, zt, 1);
}

template<class T, typename>
__device__ __host__
Matrix<4, T> TransformGen::Ortogonal(T width, T height,
                                     T nearPlane, T farPlane)
{
    //  orto    0       0       0
    //  0       orto    0       0
    //  0       0       orto    0
    //  0       0       orto    1
    T zt = nearPlane / (nearPlane - farPlane);
    return Matrix<4, T>(2 / width, 0, 0, 0,
                        0, 2 / height, 0, 0,
                        0, 0, 1 / (nearPlane - farPlane), 0,
                        0, 0, zt, 1);
}

template<class T, typename>
__device__ __host__
Matrix<4, T> TransformGen::LookAt(const Vector<3, T>& eyePos,
                                  const Vector<3, T>& at,
                                  const Vector<3, T>& up)
{
    // Calculate Orthogonal Vectors for this rotation
    Vector<3, T> zAxis = (eyePos - at).NormalizeSelf();
    Vector<3, T> xAxis = up.CrossProduct(zAxis).NormalizeSelf();
    Vector<3, T> yAxis = zAxis.CrossProduct(xAxis).NormalizeSelf();

    // Also Add Translation part
    return Matrix<4, T>(xAxis[0], yAxis[0], zAxis[0], 0,
                        xAxis[1], yAxis[1], zAxis[1], 0,
                        xAxis[2], yAxis[2], zAxis[2], 0,
                        -xAxis.Dot(eyePos), -yAxis.Dot(eyePos), -zAxis.Dot(eyePos), 1);
}

template<class T, typename>
__device__ __host__
void TransformGen::Space(Matrix<3, T>& m,
                         const Vector<3, T>& x,
                         const Vector<3, T>& y,
                         const Vector<3, T>& z)
{
    m = Matrix<3, T>(x[0], y[0], z[0],
                     x[1], y[1], z[1],
                     x[2], y[2], z[2]);
}

template<class T, typename>
__device__ __host__
void TransformGen::InvSpace(Matrix<3, T>& m,
                            const Vector<3, T>& x,
                            const Vector<3, T>& y,
                            const Vector<3, T>& z)
{
    m = Matrix<3, T>(x, y, z);
}