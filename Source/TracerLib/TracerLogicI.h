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

#include "RayLib/Vector.h"
#include "RayLib/Camera.h"
#include "RayLib/SceneError.h"

class GPUBaseAcceleratorI;
class GPUAcceleratorGroupI;
class GPUMaterialGroupI;
class RayMemory;
class RNGMemory;

using AcceleratorGroupMappings = std::map<uint16_t, GPUAcceleratorGroupI*>;
using MaterialGroupMappings = std::map<uint16_t, GPUMaterialGroupI*>;

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
		virtual											~TracerLogicI() = default;

		// Interface
		// Init & Load
		virtual TracerError								Initialize() = 0;		
		virtual SceneError								LoadScene(const std::string&) = 0;

		// Generate Camera Rays
		virtual void									GenerateCameraRays(RayMemory&, RNGMemory&,
																		   const CameraPerspective& camera,
																		   const uint32_t samplePerPixel,
																		   const Vector2ui& resolution,
																		   const Vector2ui& pixelStart,
																		   const Vector2ui& pixelCount) = 0;
		
		// Interface fetching for logic
		virtual GPUBaseAcceleratorI*					BaseAcelerator() = 0;
		virtual const AcceleratorGroupMappings&			AcceleratorGroups() = 0;
		virtual const MaterialGroupMappings&			MaterialGroups() = 0;

		// Returns bitrange of keys (should complement each other to 32-bit)
		virtual const Vector2i&							MaterialBitRange() const = 0;
		virtual const Vector2i&							AcceleratorBitRange() const = 0;

		// Options of the Hitman & Shademan
		virtual const HitOpts&							HitOptions() const = 0;
		virtual const ShadeOpts&						ShadeOptions() const = 0;

		// Loads/Unloads material to GPU Memory
		virtual void									LoadMaterial(int gpuId, HitKey key) = 0;
		virtual void									UnloadMaterial(int gpuId, HitKey matId) = 0;

		// Generates/Removes accelerator
		virtual void									GenerateAccelerator(HitKey key) = 0;
		virtual void									LoadAccelerator(HitKey key, const byte* data, size_t size) = 0;

		// Misc
		// Retuns "sizeof(RayAux)"
		virtual size_t									PerRayAuxDataSize() const = 0;
		virtual size_t									HitStructMaxSize() const = 0;

};