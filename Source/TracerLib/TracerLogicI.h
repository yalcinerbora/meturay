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
#include "RayLib/TracerStructs.h"

// Common Memory
class RayMemory;
class RNGMemory;
//
struct TracerError;

class GPUBaseAcceleratorI;
class GPUAcceleratorBatchI;
class GPUMaterialBatchI;

class TracerBaseLogicI
{
	public:
		virtual											~TracerBaseLogicI() = default;

		// Interface
		// Init
		virtual TracerError								Initialize() = 0;		
		
		// Generate camera rays
		virtual void									GenerateCameraRays(RayMemory&, RNGMemory&,
																		   const CameraPerspective& camera,
																		   const uint32_t samplePerPixel,
																		   const Vector2ui& resolution,
																		   const Vector2ui& pixelStart,
																		   const Vector2ui& pixelCount) = 0;
		// Custom ray generation logic
		virtual void									GenerateRays(RayMemory&, RNGMemory&,
																	 const uint32_t rayCount) = 0;


		
		// Interface fetching for logic
		virtual GPUBaseAcceleratorI&					BaseAcelerator() = 0;
		virtual const AcceleratorBatchMappings&			AcceleratorBatches() = 0;
		virtual const MaterialBatchMappings&			MaterialBatches() = 0;

		// Returns max bits of keys (for batch and id respectively)
		virtual const Vector2i							SceneMaterialMaxBits() const = 0;
		virtual const Vector2i							SceneAcceleratorMaxBits() const = 0;

		// Options of the Hitman & Shademan
		virtual const HitOpts&							HitOptions() const = 0;
		virtual const ShadeOpts&						ShadeOptions() const = 0;
	
		// Misc
		// Retuns "sizeof(RayAux)"
		virtual size_t									PerRayAuxDataSize() const = 0;
		// Return mimimum size of an arbitrary struct which holds all hit results
		virtual size_t									HitStructSize() const = 0;
		// Random seed
		virtual uint32_t								Seed() const = 0;
};