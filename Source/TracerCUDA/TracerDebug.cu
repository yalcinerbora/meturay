#include "TracerDebug.h"
#include "ImageMemory.h"
#include "DefaultLeaf.h"

#include "DTreeKC.cuh"
#include "STreeKC.cuh"
#include "PathNode.cuh"
#include "LinearBVH.cuh"

#include "ImageIO/EntryPoint.h"

namespace Debug
{
namespace Detail
{
    void OutputHitPairs(std::ostream& s, const RayId* ids, const HitKey* keys, size_t count);
}
}

void Debug::Detail::OutputHitPairs(std::ostream& s, const RayId* ids, const HitKey* keys, size_t count)
{
    // Do Sync this makes memory to be accessible from Host
    for(size_t i = 0; i < count; i++)
    {
        s << "{" << std::hex << std::setw(8) << std::setfill('0') << keys[i] << ", "
                 << std::dec << std::setw(0) << std::setfill(' ') << ids[i] << "}" << " ";
    }
}

void Debug::DumpImage(const std::string& fName,
                      const ImageMemory& iMem)
{
    CUDA_CHECK(cudaDeviceSynchronize());
    const ImageIOI& io = *ImageIOInstance();
    Vector2ui size(iMem.SegmentSize()[0],
                   iMem.SegmentSize()[1]);
    auto image = iMem.GMem<Vector4f>();

    ImageIOError e = ImageIOError::OK;
    if((e = io.WriteImage(reinterpret_cast<const Byte*>(image.gPixels),
                          size,
                          iMem.Format(), ImageType::PNG,
                          fName)) != ImageIOError::OK)
        METU_ERROR_LOG(static_cast<std::string>(e));
}

void Debug::DumpImage(const std::string& fName,
                      const Vector4* iMem,
                      const Vector2ui& resolution)
{
    const ImageIOI& io = *ImageIOInstance();
    ImageIOError e = ImageIOError::OK;
    if((e = io.WriteImage(reinterpret_cast<const Byte*>(iMem),
                          resolution,
                          PixelFormat::RGBA_FLOAT, ImageType::PNG,
                          fName)) != ImageIOError::OK)
        METU_ERROR_LOG(static_cast<std::string>(e));
}

void Debug::DumpBitmap(const std::string& fName,
                       const Byte* bits,
                       const Vector2ui& resolution)
{
    const ImageIOI& io = *ImageIOInstance();
    ImageIOError e = ImageIOError::OK;
    if((e = io.WriteBitmap(bits, resolution, ImageType::PNG, fName)) != ImageIOError::OK)
        METU_ERROR_LOG(static_cast<std::string>(e));
}

void Debug::PrintHitPairs(const RayId* ids, const HitKey* keys, size_t count)
{
    std::stringstream s;
    Detail::OutputHitPairs(s, ids, keys, count);
    METU_LOG(s.str());
}

void Debug::WriteHitPairs(const RayId* ids, const HitKey* keys, size_t count, const std::string& file)
{
    std::ofstream f(file);
    Detail::OutputHitPairs(f, ids, keys, count);
}

std::ostream& operator<<(std::ostream& stream, const Vector2ui& v)
{
    stream << std::setw(0)
        << v[0] << ", "
        << v[1];
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const Vector2ul& v)
{
    stream << std::setw(0)
        << v[0] << ", "
        << v[1];
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const Vector2f& v)
{
    stream << std::setw(0)
           << v[0] << ", "
           << v[1];
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const Vector3f& v)
{
    stream << std::setw(0)
        << v[0] << ", "
        << v[1] << ", "
        << v[2];
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const Vector4f& v)
{
    stream << std::setw(0)
        << v[0] << ", "
        << v[1] << ", "
        << v[2] << ", "
        << v[3];
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const AABB3f& aabb)
{
    stream << std::setw(0)
        << "{("
        << aabb.Min() << "), ("
        << aabb.Max()
        << ")}";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const RayGMem& r)
{
    stream << std::setw(0)
           << "{" << r.pos[0] << ", " << r.pos[1] << ", " << r.pos[2] << "} "
           << "{" << r.dir[0] << ", " << r.dir[1] << ", " << r.dir[2] << "} "
           << "{" << r.tMin << ", " << r.tMax << "}";
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const HitKey& key)
{
    stream << std::hex << std::setfill('0')
           << std::setw(HitKey::BatchBits / 4) << HitKey::FetchBatchPortion(key)
           << ":"
           << std::setw(HitKey::IdBits / 4) << HitKey::FetchIdPortion(key);
    stream << std::dec;
    return stream;
}

std::ostream& operator<<(std::ostream& stream, const DefaultLeaf& l)
{
    stream << std::setw(0)
           << "{ mat: " << l.matId << ", primId: " << l.primitiveId << "} ";
    return stream;
}

std::ostream& operator<<(std::ostream& s, const STreeNode& n)
{
    s << "C{";
    if(n.isLeaf)
    {
        s << "-, -} ";
        s << "T{" << n.index << "} ";
    }
    else
    {
        s << n.index << ", " << (n.index + 1) << "} ";
        s << "T{-} ";
    }

    //static constexpr const char* XYZ = "XYZ";
    s << "Axis {";
    s << "XYZ"[static_cast<int>(n.splitAxis)];
    s << "}";
    return s;
}

std::ostream& operator<<(std::ostream& s, const STreeGPU& n)
{
    s << "NodeCount  : " << n.nodeCount << std::endl;
    s << "Extents    : {{"
        << n.extents.Min()[0] << ", " << n.extents.Min()[1] << ", " << n.extents.Min()[2] << "}, {"
        << n.extents.Max()[0] << ", " << n.extents.Max()[1] << ", " << n.extents.Max()[2];
    s << "}}" << std::endl;
    return s;
}

std::ostream& operator<<(std::ostream& s, const PathGuidingNode& n)
{
    s << "W: { "
      << n.worldPosition[0] << ", "
      << n.worldPosition[1] << ", "
      << n.worldPosition[2] << "} "
      << "PN: {"
      << static_cast<uint32_t>(n.prevNext[0]) << ", "
      << static_cast<uint32_t>(n.prevNext[1]) << "} "
      << "R: {"
      << n.totalRadiance[0] << ", "
      << n.totalRadiance[1] << ", "
      << n.totalRadiance[2] << "} "
      << "DT: "
      << n.dataStructIndex;
    return s;
}

std::ostream& operator<<(std::ostream& s, const DTreeNode& n)
{
    constexpr uint32_t UINT32_T_MAX = std::numeric_limits<uint32_t>::max();

    s << "P{";
    if(n.parentIndex == UINT32_T_MAX) s << "-";
    else s << n.parentIndex;
    s << "} ";
    s << "C{";
    if(n.childIndices[0] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[0];
    s << ", ";
    if(n.childIndices[1] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[1];
    s << ", ";
    if(n.childIndices[2] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[2];
    s << ", ";
    if(n.childIndices[3] == UINT32_T_MAX) s << "-";
    else s << n.childIndices[3];
    s << "} ";
    s << "I{"
      << n.irradianceEstimates[0] << ", "
      << n.irradianceEstimates[1] << ", "
      << n.irradianceEstimates[2] << ", "
      << n.irradianceEstimates[3] << "}";
    return s;
}

std::ostream& operator<<(std::ostream& s, const DTreeGPU& n)
{
    s << "Irradiance  : " << n.irradiance << std::endl;
    s << "NodeCount  : " << n.nodeCount << std::endl;
    s << "SampleCount: " << n.totalSamples << std::endl;
    return s;
}