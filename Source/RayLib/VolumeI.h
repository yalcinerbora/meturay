#pragma once
/**

Volume is dense/sparse volume representation of a phenomena (i.e. smoke, water)


*/

#include "SurfaceI.h"
#include "AnimateI.h"
#include "Vector.h"



class VolumeI : public SurfaceI, public AnimateI
{
	public:
		virtual					~VolumeI() = default;

		// Interface
		virtual Vector3ui		Size() = 0;
};
