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
    X_POS = 0,
    X_NEG = 1,
    Y_POS = 2,
    Y_NEG = 3,
    Z_POS = 4,    
    Z_NEG = 5
};

template <int D>
struct TexDimType {};
template <>
struct TexDimType<1> { using type = unsigned int; };
template <>
struct TexDimType<2> { using type = Vector2ui; };
template <>
struct TexDimType<3> { using type = Vector3ui; };
template <int D>
using TexDimType_t = typename TexDimType<D>::type;


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
        
        TexDimType_t<D>             dim;
        
        InterpolationType           interpType;
        EdgeResolveType             edgeResolveType;

    protected:
    public:
        // Constructors & Destructor
                            Texture(int deviceId,
                                    InterpolationType,
                                    EdgeResolveType,
                                    bool convertSRGB,
                                    const TexDimType_t<D>& dim,
                                    int mipCount);
                            ~Texture();

        // Copy Data
        void                Copy(const Byte* sourceData,
                                 const TexDimType_t<D>& size,
                                 const TexDimType_t<D>& offset = Zero3ui,
                                 int mipLevel = 0);
        GPUFence            CopyAsync(const Byte* sourceData,
                                      const TexDimType_t<D>& size,
                                      const TexDimType_t<D>& offset = Zero3ui,
                                      int mipLevel = 0,
                                      cudaStream_t stream = nullptr);

        const TexDimType_t<D>&  Dim() const;
        InterpolationType       InterpType() const;
        EdgeResolveType         EdgeType() const;

        // Memory Migration
        void                    MigrateToOtherDevice(int deviceTo,
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

        TexDimType_t<D>             dim;
        unsigned int                length;

        InterpolationType           interpType;
        EdgeResolveType             edgeResolveType;

    protected:
    public:
        // Constructors & Destructor
                            TextureArray(int deviceId,
                                         InterpolationType,
                                         EdgeResolveType,
                                         bool convertSRGB,
                                         const TexDimType_t<D>& dim,
                                         unsigned int length,
                                         int mipCount);
                            ~TextureArray();

        // Copy Data
        void                Copy(const Byte* sourceData,
                                 const TexDimType_t<D>& size,
                                 int layer,
                                 const TexDimType_t<D>& offset = Zero3ui,
                                 int mipLevel = 0);
        GPUFence            CopyAsync(const Byte* sourceData,
                                      const TexDimType_t<D>& size,
                                      int layer,
                                      const TexDimType_t<D>& offset = Zero3ui,
                                      int mipLevel = 0,
                                      cudaStream_t stream = nullptr);

        // Accessors
        const TexDimType_t<D>&  Dim() const;
        unsigned int            Length() const;
        InterpolationType       InterpType() const;
        EdgeResolveType         EdgeType() const;

        void                    MigrateToOtherDevice(int deviceTo,
                                                     cudaStream_t stream = nullptr) override;
};

template<class T>
class TextureCube : public DeviceLocalMemoryI
{
    static_assert(is_TextureType_v<T>, "Invalid texture cube type");

    public:
        static constexpr uint32_t   CUBE_FACE_COUNT = 6;

    private:
        cudaMipmappedArray_t        data;
        cudaTextureObject_t         t;

        Vector2ui                   dim;
        InterpolationType           interpType;
        EdgeResolveType             edgeResolveType;

    protected:
    public:
                            TextureCube(int deviceId,
                                        InterpolationType,
                                        EdgeResolveType,
                                        bool convertSRGB,
                                        const Vector2ui& dim,
                                        int mipCount);
                            ~TextureCube();

        // Copy Data
        void                Copy(const Byte* sourceData,
                                 const Vector2ui& size,
                                 CubeTexSide,
                                 const Vector2ui& offset = Zero2ui,
                                 int mipLevel = 0);
        GPUFence            CopyAsync(const Byte* sourceData,
                                      const Vector2ui& size,
                                      CubeTexSide,
                                      const Vector2ui& offset = Zero2ui,
                                      int mipLevel = 0,
                                      cudaStream_t stream = nullptr);

        const Vector2ui&    Dim() const;
        InterpolationType   InterpType() const;
        EdgeResolveType     EdgeType() const;

        void                MigrateToOtherDevice(int deviceTo, cudaStream_t stream = nullptr) override;
};

// Ease of use Template Types
template<class T> using Texture1D = Texture<1, T>;
template<class T> using Texture2D = Texture<2, T>;
template<class T> using Texture3D = Texture<3, T>;

template<class T> using Texture1DArray = TextureArray<1, T>;
template<class T> using Texture2DArray = TextureArray<2, T>;

#include "Texture.hpp"

extern template class Texture<1, float>;
extern template class Texture<1, Vector2>;
extern template class Texture<1, Vector4>;
extern template class Texture<1, int>;
extern template class Texture<1, int2>;
extern template class Texture<1, int4>;
extern template class Texture<1, short>;
extern template class Texture<1, short2>;
extern template class Texture<1, short4>;
extern template class Texture<1, char>;
extern template class Texture<1, char2>;
extern template class Texture<1, char4>;
extern template class Texture<1, unsigned int>;
extern template class Texture<1, uint2>;
extern template class Texture<1, uint4>;
extern template class Texture<1, unsigned short>;
extern template class Texture<1, ushort2>;
extern template class Texture<1, ushort4>;
extern template class Texture<1, unsigned char>;
extern template class Texture<1, uchar2>;
extern template class Texture<1, uchar4>;

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

extern template class TextureArray<1, float>;
extern template class TextureArray<1, Vector2>;
extern template class TextureArray<1, Vector4>;
extern template class TextureArray<1, int>;
extern template class TextureArray<1, int2>;
extern template class TextureArray<1, int4>;
extern template class TextureArray<1, short>;
extern template class TextureArray<1, short2>;
extern template class TextureArray<1, short4>;
extern template class TextureArray<1, char>;
extern template class TextureArray<1, char2>;
extern template class TextureArray<1, char4>;
extern template class TextureArray<1, unsigned int>;
extern template class TextureArray<1, uint2>;
extern template class TextureArray<1, uint4>;
extern template class TextureArray<1, unsigned short>;
extern template class TextureArray<1, ushort2>;
extern template class TextureArray<1, ushort4>;
extern template class TextureArray<1, unsigned char>;
extern template class TextureArray<1, uchar2>;
extern template class TextureArray<1, uchar4>;

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