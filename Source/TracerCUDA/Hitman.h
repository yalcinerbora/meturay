#pragma once
/**

Manages hits

*/

#include <map>
#include <set>
#include "RayLib/DeviceMemory.h"
#include "RayLib/ArrayPortion.h"
//#include "RayLib/HitStructs.h"
#include "RayLib/Vector.h"

class GPUAcceleratorI;
class RayMemory;

using RayPartitionsAccelerator = std::set<ArrayPortion<uint16_t>>;

struct HitmanOptions
{
	 Vector2i keyBitRange;
};

static constexpr HitmanOptions DefaultHitmanOptions =
{
	Vector2i(24, 32)
};

class Hitman
{
	private:				
		HitmanOptions							opts;

		GPUAcceleratorI*						baseAccelerator;
		std::map<uint16_t, GPUAcceleratorI*>	subAccelerators;

		// Internal
		RayPartitionsAccelerator				Partition(uint32_t& rayCount);

	protected:
	public:
		// Constructors & Destructor
										Hitman(const HitmanOptions&);
										Hitman(const Hitman&) = delete;
										Hitman(Hitman&&) = default;
		Hitman&							operator=(const Hitman&) = delete;
		Hitman&							operator=(Hitman&&) = default;
										~Hitman() = default;


		// Interface
		void							Process(RayMemory& memory,
												uint32_t rayCount);
};
