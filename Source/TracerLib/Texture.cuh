#pragma once

/**

Lightweight texture wrapper for cuda

Object oriented design and openGL like access

*/

#include "RayLib/CudaCheck.h"
#include "RayLib/Vector.h"
#include "RayLib/Types.h"
#include "RayLib/TypeTraits.h"

#include "GPUEvent.h"
#include "DeviceMemory.h"

#include <cuda_runtime.h>
#include <cstddef>

enum class InterpolationType
{
    NEAREST,
    LINEAR
};

enum class EdgeResolveType
{
    WRAP,
    CLAMP,
    MIRROR
    // Border does not work properly
};

enum class CubeTexSide
{
    X_POS,
    Y_POS,
    Z_POS,
    X_NEG,
    Y_NEG,
    Z_NEG
};

template <class T>
struct is_TextureNormalizedType
{
    static constexpr bool value = is_any <T,
                                          char, char2, char4,
                                          short, short2, short4,
                                          int, int2, int4,
                                          unsigned char, uchar2, uchar4,
                                          unsigned short, ushort2, ushort4,
                                          unsigned int, uint2, uint4>::value;

};

template <class T>
struct is_TextureType
{
    static constexpr bool value = is_any <T,
                                          float, Vector2f, Vector4f,
                                          char, char2, char4,
                                          short, short2, short4,
                                          int, int2, int4,
                                          unsigned char, uchar2, uchar4,
                                          unsigned short, ushort2, ushort4,
                                          unsigned int, uint2, uint4>::value;
};

template <class T>
inline constexpr bool is_TextureNormalizedType_v = is_TextureNormalizedType<T>::value;

template <class T>
inline constexpr bool is_TextureType_v = is_TextureType<T>::value;

static constexpr cudaTextureAddressMode DetermineAddressMode(EdgeResolveType);
static constexpr cudaTextureFilterMode DetermineFilterMode(InterpolationType);

template<int D, class T>
class Texture 
    : public DeviceLocalMemoryI
{
    static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");
    static_assert(is_TextureType_v<T>, "Invalid texture type");

    private:
        cudaMipmappedArray_t        data;
        cudaTextureObject_t         t;
        
        Vector<D, unsigned int>     dim;
        
        InterpolationType           interpType;
        EdgeResolveType             edgeReolveType;

    protected:
    public:
        // Constructors & Destructor
                            Texture(int deviceId,
                                    InterpolationType,
                                    EdgeResolveType,
                                    const Vector<D, unsigned int>& dim);
                            ~Texture();

        // Copy Data
        void                Copy(const Byte* sourceData,
                                 const Vector<D, unsigned int>& size,
                                 const Vector<D, unsigned int>& offset = Zero3ui,
                                 int mipLevel = 0);
        GPUFence            CopyAsync(const Byte* sourceData,
                                      const Vector<D, unsigned int>& size,
                                      const Vector<D, unsigned int>& offset = Zero3ui,
                                      int mipLevel = 0,
                                      cudaStream_t stream = nullptr);

        // Memory Migration
        void                MigrateToOtherDevice(int deviceTo, 
                                                 cudaStream_t stream = nullptr) override;
};

template<int D, class T>
class TextureArray : public DeviceLocalMemoryI
{    
    static_assert(D >= 1 && D <= 2, "At most 2D texture arrays are supported");
    static_assert(is_TextureType_v<T>, "Invalid texture array type");

    private:
        cudaMipmappedArray_t        data;
        cudaTextureObject_t         t;

        Vector<D, unsigned int>     dim;
        unsigned int                count;

        InterpolationType           interpType;
        EdgeResolveType             edgeReolveType;

    protected:
    public:
        // Constructors & Destructor
                            TextureArray(int deviceId, 
                                         InterpolationType,
                                         EdgeResolveType,
                                         const Vector<D, unsigned int>& dim,
                                         unsigned int count);
                            ~TextureArray();

        // Copy Data
        void                Copy(const Byte* sourceData,
                                 const Vector<D, unsigned int>& size,
                                 int layer,
                                 const Vector<D, unsigned int>& offset = Zero3ui,
                                 int mipLevel = 0);
        GPUFence            CopyAsync(const Byte* sourceData,
                                      const Vector<D, unsigned int>& size,
                                      int layer,
                                      const Vector<D, unsigned int>& offset = Zero3ui,
                                      int mipLevel = 0,
                                      cudaStream_t stream = nullptr);

        void                MigrateToOtherDevice(int deviceTo,
                                                 cudaStream_t stream = nullptr) override;
};

template<class T>
class TextureCube : public DeviceLocalMemoryI
{
    static_assert(is_TextureType_v<T>, "Invalid texture cube type");

    private:
        cudaArray_t           data;
        cudaTextureObject_t   t;

        Vector2ui             dim;
        InterpolationType     interpType;
        EdgeResolveType       edgeResolve;

    protected:
    public:
                            TextureCube(int deviceId,
                                        InterpolationType,
                                        EdgeResolveType,
                                        const Vector2ui& dim);
                            ~TextureCube();

        // Copy Data
        void                Copy(const Byte* sourceData,
                                 const Vector<2, unsigned int>& size,
                                 CubeTexSide,
                                 const Vector<2, unsigned int>& offset = Zero2ui,
                                 int mipLevel = 0);
        GPUFence            CopyAsync(const Byte* sourceData,
                                      const Vector<2, unsigned int>& size,
                                      CubeTexSide,
                                      const Vector<2, unsigned int>& offset = Zero2ui,
                                      int mipLevel = 0,
                                      cudaStream_t stream = nullptr);

        void                MigrateToOtherDevice(int deviceTo, cudaStream_t stream = nullptr) override;
};

// Ease of use Template Types
template<class T> using Texture1D = Texture<1, T>;
template<class T> using Texture2D = Texture<2, T>;
template<class T> using Texture3D = Texture<3, T>;

template<class T> using Texture1DArray = TextureArray<1, T>;
template<class T> using Texture2DArray = TextureArray<2, T>;

#include "Texture.hpp"

//extern template class Texture<1, float>;
//extern template class Texture<1, Vector2>;
//extern template class Texture<1, Vector4>;
//extern template class Texture<1, int>;
//extern template class Texture<1, int2>;
//extern template class Texture<1, int4>;
//extern template class Texture<1, short>;
//extern template class Texture<1, short2>;
//extern template class Texture<1, short4>;
//extern template class Texture<1, char>;
//extern template class Texture<1, char2>;
//extern template class Texture<1, char4>;
//extern template class Texture<1, unsigned int>;
//extern template class Texture<1, uint2>;
//extern template class Texture<1, uint4>;
//extern template class Texture<1, unsigned short>;
//extern template class Texture<1, ushort2>;
//extern template class Texture<1, ushort4>;
//extern template class Texture<1, unsigned char>;
//extern template class Texture<1, uchar2>;
//extern template class Texture<1, uchar4>;

extern template class Texture<2, float>;
extern template class Texture<2, Vector2>;
extern template class Texture<2, Vector4>;
extern template class Texture<2, int>;
extern template class Texture<2, int2>;
extern template class Texture<2, int4>;
extern template class Texture<2, short>;
extern template class Texture<2, short2>;
extern template class Texture<2, short4>;
extern template class Texture<2, char>;
extern template class Texture<2, char2>;
extern template class Texture<2, char4>;
extern template class Texture<2, unsigned int>;
extern template class Texture<2, uint2>;
extern template class Texture<2, uint4>;
extern template class Texture<2, unsigned short>;
extern template class Texture<2, ushort2>;
extern template class Texture<2, ushort4>;
extern template class Texture<2, unsigned char>;
extern template class Texture<2, uchar2>;
extern template class Texture<2, uchar4>;

extern template class Texture<3, float>;
extern template class Texture<3, Vector2>;
extern template class Texture<3, Vector4>;
extern template class Texture<3, int>;
extern template class Texture<3, int2>;
extern template class Texture<3, int4>;
extern template class Texture<3, short>;
extern template class Texture<3, short2>;
extern template class Texture<3, short4>;
extern template class Texture<3, char>;
extern template class Texture<3, char2>;
extern template class Texture<3, char4>;
extern template class Texture<3, unsigned int>;
extern template class Texture<3, uint2>;
extern template class Texture<3, uint4>;
extern template class Texture<3, unsigned short>;
extern template class Texture<3, ushort2>;
extern template class Texture<3, ushort4>;
extern template class Texture<3, unsigned char>;
extern template class Texture<3, uchar2>;
extern template class Texture<3, uchar4>;

//extern template class TextureArray<1, float>;
//extern template class TextureArray<1, Vector2>;
//extern template class TextureArray<1, Vector4>;
//extern template class TextureArray<1, int>;
//extern template class TextureArray<1, int2>;
//extern template class TextureArray<1, int4>;
//extern template class TextureArray<1, short>;
//extern template class TextureArray<1, short2>;
//extern template class TextureArray<1, short4>;
//extern template class TextureArray<1, char>;
//extern template class TextureArray<1, char2>;
//extern template class TextureArray<1, char4>;
//extern template class TextureArray<1, unsigned int>;
//extern template class TextureArray<1, uint2>;
//extern template class TextureArray<1, uint4>;
//extern template class TextureArray<1, unsigned short>;
//extern template class TextureArray<1, ushort2>;
//extern template class TextureArray<1, ushort4>;
//extern template class TextureArray<1, unsigned char>;
//extern template class TextureArray<1, uchar2>;
//extern template class TextureArray<1, uchar4>;

extern template class TextureArray<2, float>;
extern template class TextureArray<2, Vector2>;
extern template class TextureArray<2, Vector4>;
extern template class TextureArray<2, int>;
extern template class TextureArray<2, int2>;
extern template class TextureArray<2, int4>;
extern template class TextureArray<2, short>;
extern template class TextureArray<2, short2>;
extern template class TextureArray<2, short4>;
extern template class TextureArray<2, char>;
extern template class TextureArray<2, char2>;
extern template class TextureArray<2, char4>;
extern template class TextureArray<2, unsigned int>;
extern template class TextureArray<2, uint2>;
extern template class TextureArray<2, uint4>;
extern template class TextureArray<2, unsigned short>;
extern template class TextureArray<2, ushort2>;
extern template class TextureArray<2, ushort4>;
extern template class TextureArray<2, unsigned char>;
extern template class TextureArray<2, uchar2>;
extern template class TextureArray<2, uchar4>;

extern template class TextureCube<float>;
extern template class TextureCube<Vector2>;
extern template class TextureCube<Vector4>;
extern template class TextureCube<int>;
extern template class TextureCube<int2>;
extern template class TextureCube<int4>;
extern template class TextureCube<short>;
extern template class TextureCube<short2>;
extern template class TextureCube<short4>;
extern template class TextureCube<char>;
extern template class TextureCube<char2>;
extern template class TextureCube<char4>;
extern template class TextureCube<unsigned int>;
extern template class TextureCube<uint2>;
extern template class TextureCube<uint4>;
extern template class TextureCube<unsigned short>;
extern template class TextureCube<ushort2>;
extern template class TextureCube<ushort4>;
extern template class TextureCube<unsigned char>;
extern template class TextureCube<uchar2>;
extern template class TextureCube<uchar4>;