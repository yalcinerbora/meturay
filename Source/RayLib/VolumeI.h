#pragma once
/**

Volume is dense/sparse volume representation of a phenomena (i.e. smoke, water)


*/

#include <string>

#include "SurfaceI.h"
#include "AnimateI.h"
#include "Vector.h"

struct Error;

enum class VolumeType
{
	MAYA_NCACHE_FLUID,

	END
};

static constexpr size_t VolumeTypeSize = static_cast<size_t>(VolumeType::END);

class VolumeI : public SurfaceI, public AnimateI
{
	public:
		virtual						~VolumeI() = default;

		// Interface
		virtual Vector3ui			Size() const = 0;
		virtual const std::string&	FileName() const = 0;

		virtual Error				Load() = 0;
};