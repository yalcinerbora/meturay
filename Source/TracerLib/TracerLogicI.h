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

// Execution Related Abstraction
class GPUBaseAcceleratorI;
class GPUAcceleratorBatchI;
class GPUMaterialBatchI;
// Data Related Abstraction
class GPUPrimitiveGroupI;
class GPUAcceleratorGroupI;
class GPUMaterialGroupI;
// Common Memory
class RayMemory;
class RNGMemory;

using AcceleratorBatchMappings = std::map<uint32_t, GPUAcceleratorBatchI*>;
using MaterialBatchMappings = std::map<uint32_t, GPUMaterialBatchI*>;

struct ShadeOpts
{
	int i;
};

struct HitOpts
{
	int j;
};

struct TracerOptions
{
	Vector2i		materialKeyRange;
	Vector2i		acceleratorKeyRange;

	uint32_t		seed;
};

class TracerBaseLogicI
{
	public:
		virtual											~TracerBaseLogicI() = default;

		// Interface
		// Init
		virtual TracerError								Initialize() = 0;		
		

		// Generate Camera Rays
		virtual void									GenerateCameraRays(RayMemory&, RNGMemory&,
																		   const CameraPerspective& camera,
																		   const uint32_t samplePerPixel,
																		   const Vector2ui& resolution,
																		   const Vector2ui& pixelStart,
																		   const Vector2ui& pixelCount) = 0;
		
		// Interface fetching for logic
		virtual GPUBaseAcceleratorI*					BaseAcelerator() = 0;
		virtual const AcceleratorBatchMappings&			AcceleratorBatches() = 0;
		virtual const MaterialBatchMappings&			MaterialBatches() = 0;

		// Returns bitrange of keys (should complement each other to 32-bit)
		virtual const Vector2i							SceneMaterialMaxBits() const = 0;
		virtual const Vector2i							SceneAcceleratorMaxBits() const = 0;

		// Options of the Hitman & Shademan
		virtual const HitOpts&							HitOptions() const = 0;
		virtual const ShadeOpts&						ShadeOptions() const = 0;

		//// Loads/Unloads material to GPU Memory
		//virtual SceneError							LoadScene(const std::string&) = 0;
		//virtual void									LoadMaterial(int gpuId, HitKey key) = 0;
		//virtual void									UnloadMaterial(int gpuId, HitKey matId) = 0;

		//// Generates/Removes accelerator
		//virtual void									GenerateAccelerator(HitKey key) = 0;
		//virtual void									LoadAccelerator(HitKey key, const Byte* data, size_t size) = 0;

		// Misc
		// Retuns "sizeof(RayAux)"
		virtual size_t									PerRayAuxDataSize() const = 0;
		// Return mimimum size of an arbitrary struct which holds all hit results
		virtual size_t									HitStructSize() const = 0;
};


class TracerLogicGeneratorI
{
	public:
	virtual								~TracerLogicGeneratorI() = default;

	// Logic Generators
	virtual SceneError					GetPrimitiveGroup(GPUPrimitiveGroupI*&,
														  const std::string& primitiveType) = 0;
	virtual SceneError					GetAcceleratorGroup(GPUAcceleratorGroupI*&,
															const GPUPrimitiveGroupI&,
															const std::string& accelType) = 0;
	virtual SceneError					GetMaterialGroup(GPUMaterialGroupI*&,
														 const GPUPrimitiveGroupI&,
														 const std::string& materialType) = 0;

	virtual size_t						CurrentMinHitSize() const = 0;
};