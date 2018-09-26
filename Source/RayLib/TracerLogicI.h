#pragma once
/**

Tracer Logic:

Responsible for containing logic CUDA Tracer

This wll be wrapped by a template class which partially implements
some portions of the main code

That interface is responsible for fetching


*/

#include <cstdint>
#include <map>

#include "Vector.h"
#include "Camera.h"

class GPUBaseAcceleratorI;
class GPUAcceleratorI;
class GPUMaterialI;
class RayMemory;
class RNGMemory;

struct ShadeOpts
{
	int i;
};

struct HitOpts
{
	int j;
};

class TracerLogicI
{
	public:
		virtual												~TracerLogicI() = default;


		virtual TracerError									Initialize() = 0;

		// Generate Camera Rays
		virtual void										GenerateCameraRays(RayMemory&, RNGMemory&,
																			   const CameraPerspective& camera,
																			   const uint32_t samplePerPixel,
																			   const Vector2ui& resolution,
																			   const Vector2ui& pixelStart,
																			   const Vector2ui& pixelCount) = 0;

		// Accessors for Managers
		// Hitman is responsible for
		virtual const std::string&								HitmanName() const = 0;
		virtual const std::string&								ShademanName() const = 0;

		// Interface fetching for logic
		virtual GPUBaseAcceleratorI*							BaseAcelerator() = 0;
		virtual const std::map<uint16_t, GPUAcceleratorI*>&		Accelerators() = 0;
		virtual const std::map<uint32_t, GPUMaterialI*>&		Materials() = 0;

		// Returns bitrange of keys (should complement each other to 32-bit)
		virtual const Vector2i&									MaterialBitRange() const = 0;
		virtual const Vector2i&									AcceleratorBitRange() const = 0;

		// Options of the Hitman & Shademan
		virtual const HitOpts&									HitOptions() const = 0;
		virtual const ShadeOpts&								ShadeOptions() const = 0;

		// Loads/Unloads material to GPU Memory
		virtual void											LoadMaterial(int gpuId, uint32_t matId) = 0;
		virtual void											UnloadMaterial(int gpuId, uint32_t matId) = 0;

		// Generates/Removes accelerator
		virtual void											GenerateAccelerator(uint32_t accId) = 0;
		virtual void											RemoveAccelerator(uint32_t accId) = 0;

		// Misc
		// Retuns "sizeof(RayAux)"
		virtual size_t											PerRayAuxDataSize() = 0;
};