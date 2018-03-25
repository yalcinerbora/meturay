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
		Vector3i						dim;
		Vector3f						size;	
		// Color Interpolation
		std::vector<Vector3f>			color;
		std::vector<float>				colorInterp;
		// Opacity Interpolation
		std::vector<float>				opacity;
		std::vector<float>				opacityInterp;
		// Transparency
		Vector3f						transparency;
		// Channels
		std::vector<MayaChannelType>	channels;
	};

	IOError			LoadNCacheNavierStokesXML(MayaNSCacheInfo&,
											  const std::string& fileName);
	IOError			LoadNCacheNavierStokes(std::vector<float>& densityData,
										   std::vector<float>& velocityData,
										   const MayaNSCacheInfo&,
										   const std::string& fileName);
};