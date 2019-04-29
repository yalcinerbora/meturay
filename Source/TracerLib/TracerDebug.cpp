#include "TracerDebug.h"
#include "ImageMemory.h"

#include "RayLib/ImageIO.h"

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
    CUDA_CHECK(cudaDeviceSynchronize());
    for(size_t i = 0; i < count; i++)
    {
        s << "{" << std::hex << std::setw(8) << std::setfill('0') << keys[i] << ", "
                 << std::dec << std::setw(0) << std::setfill(' ') << ids[i] << "}" << " ";
    }
}

void Debug::DumpImage(const std::string& fName,
                      const ImageMemory& iMem)
{
    ImageIO io;
    Vector2ui size(iMem.SegmentSize()[0],
                   iMem.SegmentSize()[1]);

    std::vector<Vector4f> image = iMem.MoveImageToCPU<Vector4f>();
    io.WriteAsPNG(image.data(), size, fName);
}

void Debug::PrintHitPairs(const RayId* ids, const HitKey* keys, size_t count)
{
    std::stringstream s;
    Detail::OutputHitPairs(s, ids, keys, count);
    METU_LOG("%s", s.str().c_str());
}

void Debug::WriteHitPairs(const RayId* ids, const HitKey* keys, size_t count, const std::string& file)
{
    std::ofstream f(file);
    Detail::OutputHitPairs(f, ids, keys, count);
}

std::ostream& operator<<(std::ostream& stream, const Vector2f& v)
{
    stream << std::setw(0)
           << v[0] << ", "
           << v[1];
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