#pragma once
/**

Maya nCache File I-O

Reads are not universal. Specific options are required

*/

#include <vector>
#include "Vector.h"
#include "IOError.h"

namespace MayaCache
{
    constexpr const char* FrameTag = "Frame";
    constexpr const char* Extension = "mcx";

    enum MayaChannelType
    {
        DENSITY,
        VELOCITY,
        RESOLUTION,
        OFFSET
    };

    struct MayaNSCacheInfo
    {
        Vector3ui                       dim;
        Vector3f                        size;
        // Channels
        std::vector<MayaChannelType>    channels;

        // Color Interpolation
        std::vector<Vector3f>           color;
        std::vector<float>              colorInterp;
        // Opacity Interpolation
        std::vector<float>              opacity;
        std::vector<float>              opacityInterp;
        // Transparency
        Vector3f                        transparency;
    };

    std::u8string   GenerateNCacheFrameFile(const std::u8string& xmlFile, int frame);

    IOError         LoadNCacheNavierStokesXML(MayaNSCacheInfo&,
                                              const std::u8string& fileName);
    IOError         LoadNCacheNavierStokes(std::vector<float>& velocityDensityData,
                                           const MayaNSCacheInfo&,
                                           const std::u8string& fileName);
}