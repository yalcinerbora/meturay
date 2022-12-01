#pragma once

/**

Lightweight texture wrapper for cuda

Object oriented design and openGL like access

*/

#include "RayLib/CudaCheck.h"
#include "RayLib/Vector.h"
#include "RayLib/Types.h"
#include "RayLib/TypeTraits.h"
#include "RayLib/BitManipulation.h"

#include "GPUEvent.h"
#include "DeviceMemory.h"
#include "TextureReference.cuh"
#include "CudaSystem.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

template <class T>
struct ChannelDescType { using type = T; };
template <>
struct ChannelDescType<Vector2> { using type = float2; };
template <>
struct ChannelDescType<Vector4> { using type = float4; };
template <class T>
using ChannelDescType_t = typename ChannelDescType<T>::type;

template <int D>
struct TexDimType {};
template <>
struct TexDimType<1> { using type = uint32_t; static constexpr type ZERO = 0; };
template <>
struct TexDimType<2> { using type = Vector2ui; static constexpr type ZERO = Zero2ui; };
template <>
struct TexDimType<3> { using type = Vector3ui; static constexpr type ZERO = Zero3ui; };
template <int D>
using TexDimType_t = typename TexDimType<D>::type;

template <class T>
struct TextureChannelCount
{
    private:
        static constexpr int SelectSize()
        {
            if constexpr(is_any<T,
                                float,
                                char, short, int,
                                unsigned char,
                                unsigned short,
                                unsigned int>::value)
                return 1;
            else if constexpr(is_any<T,
                                     Vector2,
                                     char2, short2, int2,
                                     uchar2, ushort2, uint2>::value)
                return 2;
            else if constexpr(is_any<T,
                                     Vector4,
                                     char4, short4, int4,
                                     uchar4, ushort4, uint4>::value)
                return 4;

            return 0;
        }

    public:
        static constexpr int value = SelectSize();
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
                                          unsigned int, uint2, uint4,
                                          half, half2 /*half4*/>::value;
};

template <class T>
constexpr bool is_TextureType_v = is_TextureType<T>::value;

static constexpr cudaTextureAddressMode DetermineAddressMode(EdgeResolveType);
static constexpr cudaTextureFilterMode DetermineFilterMode(InterpolationType);

// RAII Wrapper for surface type
// used while constructing mipmaps
class CudaSurfaceRAII
{
    private:
    cudaSurfaceObject_t surface;// = 0;

    public:
    // Constructors & Destructor
    //CudaSurfaceRAII() = default;
    CudaSurfaceRAII(cudaSurfaceObject_t);
    CudaSurfaceRAII(const CudaSurfaceRAII&) = delete;
    CudaSurfaceRAII(CudaSurfaceRAII&&);
    CudaSurfaceRAII& operator=(const CudaSurfaceRAII&) = delete;
    CudaSurfaceRAII& operator=(CudaSurfaceRAII&&);
    ~CudaSurfaceRAII();

    operator cudaSurfaceObject_t() { return surface; }
};

// Generic Texture Type
// used to group different of textures
template<int D>
class TextureI : public DeviceLocalMemoryI
{
    private:
        static constexpr uint32_t   DimensionCount  = D;

    protected:
        uint32_t                    channelCount    = 0;
        cudaTextureObject_t         texture         = 0;
        TexDimType_t<D>             dimensions      = TexDimType<D>::ZERO;
        uint32_t                    mipCount        = 0;

    public:
        // Constructors & Destructor
                                    TextureI(const TexDimType_t<D>& dim,
                                             uint32_t channelCount,
                                             const CudaGPU* device,
                                             uint32_t mipCount);
                                    TextureI(const TextureI&) = delete;
                                    TextureI(TextureI&&);
        TextureI&                   operator=(const TextureI&) = delete;
        TextureI&                   operator=(TextureI&&);
        virtual                     ~TextureI() = default;

        const TexDimType_t<D>&      Dimensions() const;
        uint32_t                    ChannelCount() const;
        uint32_t                    MipmapCount() const;

        constexpr explicit          operator cudaTextureObject_t() const;
};

template<int D>
class TextureArrayI : public DeviceLocalMemoryI
{
    private:
        static constexpr uint32_t   DimensionCount  = D;

    protected:
        cudaTextureObject_t         texture         = 0;
        uint32_t                    channelCount    = 0;
        TexDimType_t<D>             dimensions      = TexDimType<D>::ZERO;
        uint32_t                    length          = 0;

    public:
        // Constructors & Destructor
                                    TextureArrayI(const TexDimType_t<D>& dim,
                                                  uint32_t channelCount,
                                                  uint32_t layerCount,
                                                  const CudaGPU* device);
                                    TextureArrayI(const TextureArrayI&) = delete;
                                    TextureArrayI(TextureArrayI&&);
        TextureArrayI&              operator=(const TextureArrayI&) = delete;
        TextureArrayI&              operator=(TextureArrayI&&);
        virtual                     ~TextureArrayI() = default;


        const TexDimType_t<D>&      Dimensions() const;
        uint32_t                    Length() const;
        uint32_t                    ChannelCount() const;

        constexpr explicit          operator cudaTextureObject_t() const;
};

class TextureCubeI : public DeviceLocalMemoryI
{
    private:
        static constexpr uint32_t   DimensionCount  = 2;
        static constexpr uint32_t   CubeSideCount   = 6;

    protected:
        uint32_t                    channelCount    = 0;
        cudaTextureObject_t         texture         = 0;
        Vector2ui                   dimensions      = Zero2ui;

    public:
        // Constructors & Destructor
                                    TextureCubeI(const Vector2ui& dim,
                                                 uint32_t channelCount,
                                                 const CudaGPU* device);
                                    TextureCubeI(const TextureCubeI&) = delete;
                                    TextureCubeI(TextureCubeI&&);
        TextureCubeI&               operator=(const TextureCubeI&) = delete;
        TextureCubeI&               operator=(TextureCubeI&&);
        virtual                     ~TextureCubeI() = default;

        const Vector2ui&            Dimensions() const;
        uint32_t                    ChannelCount() const;

        constexpr explicit          operator cudaTextureObject_t() const;
};

template<int D, class T>
class Texture final : public TextureI<D>
{
    static_assert(D >= 1 && D <= 3, "At most 3D textures are supported");
    static_assert(is_TextureType_v<T>, "Invalid texture type");

    public:
        using ChannelType = T;


    private:
        cudaMipmappedArray_t        data = nullptr;


        InterpolationType           interpType;
        EdgeResolveType             edgeResolveType;
        bool                        normalizeIntegers;
        bool                        normalizeCoordinates;
        bool                        convertSRGB;

    protected:
    public:
        // Constructors & Destructor
                            Texture() = delete;
                            Texture(const CudaGPU* device,
                                    InterpolationType,
                                    EdgeResolveType,
                                    bool normalizeIntegers,
                                    bool normalizeCoordinates,
                                    bool convertSRGB,
                                    const TexDimType_t<D>& dim,
                                    uint32_t mipCount);
                            Texture(const Texture&) = delete;
                            Texture(Texture&&);
        Texture&            operator=(const Texture&) = delete;
        Texture&            operator=(Texture&&);
                            ~Texture();

        // Copy Data
        void                Copy(const Byte* sourceData,
                                 const TexDimType_t<D>& size,
                                 const TexDimType_t<D>& offset = TexDimType<D>::ZERO,
                                 uint32_t mipLevel = 0);
        GPUFence            CopyAsync(const Byte* sourceData,
                                      const TexDimType_t<D>& size,
                                      const TexDimType_t<D>& offset = TexDimType<D>::ZERO,
                                      uint32_t mipLevel = 0,
                                      cudaStream_t stream = nullptr);

        // Generates empty mipmapped texture
        // does not generate any mipmaps,
        // copies the level zero to the created texture though
        Texture<D, T>           EmptyMipmappedTexture(uint32_t upToLevel = std::numeric_limits<uint32_t>::max()) const;
        CudaSurfaceRAII         GetMipLevelSurface(uint32_t level);
        void                    GetRawPixelData(std::vector<Byte>& hPixels,
                                                uint32_t mipLevel) const;

        // Accessors
        InterpolationType       InterpType() const;
        EdgeResolveType         EdgeType() const;

        // Misc
        size_t                  Size() const override;

        // Memory Migration
        void                    MigrateToOtherDevice(const CudaGPU* deviceTo,
                                                     cudaStream_t stream = nullptr) override;
};

template<int D, class T>
class TextureArray final : public TextureArrayI<D>
{
    static_assert(D >= 1 && D <= 2, "At most 2D texture arrays are supported");
    static_assert(is_TextureType_v<T>, "Invalid texture array type");

    private:
        cudaMipmappedArray_t        data    = nullptr;
        cudaTextureObject_t         texture = 0;


        InterpolationType           interpType;
        EdgeResolveType             edgeResolveType;

    protected:
    public:
        // Constructors & Destructor
                            TextureArray() = delete;
                            TextureArray(const CudaGPU* device,
                                         InterpolationType,
                                         EdgeResolveType,
                                         bool normalizeIntegers,
                                         bool normalizeCoordinates,
                                         bool convertSRGB,
                                         const TexDimType_t<D>& dim,
                                         unsigned int length,
                                         int mipCount);
                            TextureArray(const TextureArray&) = delete;
                            TextureArray(TextureArray&&);
        TextureArray&       operator=(const TextureArray&) = delete;
        TextureArray&       operator=(TextureArray&&);
                            ~TextureArray();

        // Copy Data
        void                Copy(const Byte* sourceData,
                                 const TexDimType_t<D>& size,
                                 int layer,
                                 const TexDimType_t<D>& offset = TexDimType<D>::ZERO,
                                 int mipLevel = 0);
        GPUFence            CopyAsync(const Byte* sourceData,
                                      const TexDimType_t<D>& size,
                                      int layer,
                                      const TexDimType_t<D>& offset = TexDimType<D>::ZERO,
                                      int mipLevel = 0,
                                      cudaStream_t stream = nullptr);

        // Accessors
        InterpolationType       InterpType() const;
        EdgeResolveType         EdgeType() const;

        // Misc
        size_t                  Size() const override;

        void                    MigrateToOtherDevice(const CudaGPU* deviceTo,
                                                     cudaStream_t stream = nullptr) override;
};

template<class T>
class TextureCube final : public TextureCubeI
{
    static_assert(is_TextureType_v<T>, "Invalid texture cube type");

    public:
        static constexpr uint32_t   CUBE_FACE_COUNT = 6;

    private:
        cudaMipmappedArray_t        data    = nullptr;
        cudaTextureObject_t         texture = 0;
        Vector2ui                   dim     = Zero3ui;

        InterpolationType           interpType;
        EdgeResolveType             edgeResolveType;

    protected:
    public:
        // Constructors & Destructor
                            TextureCube() = delete;
                            TextureCube(const CudaGPU* device,
                                        InterpolationType,
                                        EdgeResolveType,
                                        bool normalizeIntegers,
                                        bool convertSRGB,
                                        const Vector2ui& dim,
                                        int mipCount);
                            TextureCube(const TextureCube&) = delete;
                            TextureCube(TextureCube&&);
        TextureCube&        operator=(const TextureCube&) = delete;
        TextureCube&        operator=(TextureCube&&);
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

        InterpolationType   InterpType() const;
        EdgeResolveType     EdgeType() const;

        // Misc
        size_t              Size() const override;

        void                MigrateToOtherDevice(const CudaGPU* deviceTo, cudaStream_t stream = nullptr) override;
};

// Ease of use Template Types
template<class T> using Texture1D = Texture<1, T>;
template<class T> using Texture2D = Texture<2, T>;
template<class T> using Texture3D = Texture<3, T>;

template<class T> using Texture1DArray = TextureArray<1, T>;
template<class T> using Texture2DArray = TextureArray<2, T>;

inline CudaSurfaceRAII::CudaSurfaceRAII(cudaSurfaceObject_t s)
 : surface(s)
{}

inline CudaSurfaceRAII::CudaSurfaceRAII(CudaSurfaceRAII&& other)
    : surface(other.surface)
{
    other.surface = 0;
}

inline CudaSurfaceRAII& CudaSurfaceRAII::operator=(CudaSurfaceRAII&& other)
{
    if(surface) CUDA_CHECK(cudaDestroySurfaceObject(surface));
    surface = other.surface;
    other.surface = 0;
}

inline CudaSurfaceRAII::~CudaSurfaceRAII()
{
    if(surface) CUDA_CHECK(cudaDestroySurfaceObject(surface));
}

template<int D>
inline TextureI<D>::TextureI(const TexDimType_t<D>& dim,
                             uint32_t channelCount,
                             const CudaGPU* currentDevice,
                             uint32_t mipCount)
    : DeviceLocalMemoryI(currentDevice)
    , channelCount(channelCount)
    , dimensions(dim)
    , mipCount(mipCount)
{}

template<int D>
constexpr TextureI<D>::operator cudaTextureObject_t() const
{
    return texture;
}

template<int D>
const TexDimType_t<D>& TextureI<D>::Dimensions() const
{
    return dimensions;
}

template<int D>
uint32_t TextureI<D>::ChannelCount() const
{
    return channelCount;
}

template<int D>
uint32_t TextureI<D>::MipmapCount() const
{
    return mipCount;
}

template<int D>
inline TextureArrayI<D>::TextureArrayI(const TexDimType_t<D>& dim,
                                       uint32_t channelCount,
                                       uint32_t length,
                                       const CudaGPU* currentDevice)
    : DeviceLocalMemoryI(currentDevice)
    , channelCount(channelCount)
    , dimensions(dim)
    , length(length)
{}

template<int D>
constexpr TextureArrayI<D>::operator cudaTextureObject_t() const
{
    return texture;
}

template<int D>
const TexDimType_t<D>& TextureArrayI<D>::Dimensions() const
{
    return dimensions;
}

template<int D>
uint32_t TextureArrayI<D>::Length() const
{
    return length;
}

template<int D>
uint32_t TextureArrayI<D>::ChannelCount() const
{
    return channelCount;
}

inline TextureCubeI::TextureCubeI(const Vector2ui& dim,
                                  uint32_t channelCount,
                                  const CudaGPU* currentDevice)
    : DeviceLocalMemoryI(currentDevice)
    , channelCount(channelCount)
    , dimensions(dim)
{}

constexpr TextureCubeI::operator cudaTextureObject_t() const
{
    return texture;
}

inline const Vector2ui& TextureCubeI::Dimensions() const
{
    return dimensions;
}

inline uint32_t TextureCubeI::ChannelCount() const
{
    return channelCount;
}

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