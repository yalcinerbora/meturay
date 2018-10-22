#pragma once

#include "GPUAcceleratorI.h"

template <class PGroup>
class GPUAcceleratorGroup : public GPUAcceleratorGroupI
{
	public:
		using LeafStruct				= PGroup::LeafStruct;
	
	private:
	protected:
		// From Tracer
		const PGroup&						primitiveGroup;
		const TransformStruct*				dInverseTransforms;

		// CPU Memory
		std::map<uint32_t, SurfaceMaterialPairs>	acceleratorData;

	public:
};