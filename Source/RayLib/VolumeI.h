#pragma once
/**

Volume is dense/sparse volume representation of a phenomena (i.e. smoke, water)


*/

#include <string>

#include "SurfaceI.h"
#include "AnimateI.h"
#include "Vector.h"
#include "IOError.h"

enum class VolumeType
{
	MAYA_NCACHE_FLUID,

	END
};

static constexpr size_t VolumeTypeSize = static_cast<size_t>(VolumeType::END);

class VolumeI : public SurfaceI, public AnimateI
{
	public:
		virtual					~VolumeI() = default;

		// Interface
		virtual Vector3ui		Size() = 0;

		virtual IOError			Load(const std::string& fileName, VolumeType) = 0;
};