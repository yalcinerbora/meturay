#pragma once

#include "GPUEndpointI.h"

#include "RayLib/Constants.h"
#include "RayLib/SceneStructs.h"
#include "CudaConstants.h"

#include "NodeListing.h"

#include <type_traits>

class GPUTransformI;
class RandomGPU;

using GPULightI = GPUEndpointI;
using GPULightList = std::vector<const GPULightI*>;

class CPULightGroupI
{
	protected:
		// Global Transform Array
		const GPUTransformI** dGPUTransforms;
	public:
										CPULightGroupI();
		virtual							~CPULightGroupI() = default;

		// Interface
		virtual const char*				Type() const = 0;
		virtual const GPULightList&		GPULights() const = 0;
		virtual SceneError				InitializeGroup(const ConstructionDataList& lightNodes,
                                                        const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
                                                        const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
                                                        const MaterialKeyListing& allMaterialKeys,
														double time,
														const std::string& scenePath) = 0;
		virtual SceneError				ChangeTime(const NodeListing& lightNodes, double time,
												   const std::string& scenePath) = 0;
		virtual TracerError				ConstructLights(const CudaSystem&,
														const GPUTransformI**) = 0;
		virtual uint32_t				LightCount() const = 0;

		virtual size_t					UsedGPUMemory() const = 0;
		virtual size_t					UsedCPUMemory() const = 0;
};

inline CPULightGroupI::CPULightGroupI()
	: dGPUTransforms(nullptr)
{}