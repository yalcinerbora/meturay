#pragma once

#include "GPUEndpointI.h"

#include "RayLib/Constants.h"
#include "RayLib/SceneStructs.h"
#include "CudaSystem.h"

#include "NodeListing.h"

#include <type_traits>

class GPUBoundaryMaterialGroupI;
class GPUTransformI;
class RandomGPU;

using GPULightI = GPUEndpointI;
using GPULightList = std::vector<const GPULightI*>;

using KeyMaterialMap = std::map<uint32_t, const GPUBoundaryMaterialGroupI*>;

class CPULightGroupI
{
	public:
		virtual							~CPULightGroupI() = default;

		// Interface
		virtual const char*						Type() const = 0;
		virtual const GPULightList&				GPULights() const = 0;
		virtual SceneError						InitializeGroup(const LightGroupDataList& lightNodes,
												                const std::map<uint32_t, uint32_t>& mediumIdIndexPairs,
												                const std::map<uint32_t, uint32_t>& transformIdIndexPairs,
												                const MaterialKeyListing& allMaterialKeys,
																double time,
																const std::string& scenePath) = 0;
		virtual SceneError						ChangeTime(const NodeListing& lightNodes, double time,
														   const std::string& scenePath) = 0;
		virtual TracerError						ConstructLights(const CudaSystem&,
																const GPUTransformI**,
																const KeyMaterialMap&) = 0;
		virtual uint32_t						LightCount() const = 0;

		virtual size_t							UsedGPUMemory() const = 0;
		virtual size_t							UsedCPUMemory() const = 0;
};