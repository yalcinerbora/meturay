#pragma once



#include "TracerLogicI.h"
#include "GPUPrimitiveI.h"

struct TracerError;

template<class RayAuxData>
using AuxInitFunc = void(*)(const RayAuxData*,
							const uint32_t writeLoc,
							// Input
							const RayAuxData,
							// Index
							const Vector2ui& globalPixelId,
							const Vector2ui& localSampleId,
							const uint32_t samplePerPixel);

template<class RayAuxD, AuxInitFunc<RayAuxD> AuxFunc>
class TracerBaseLogic : public TracerBaseLogicI
{
	public:
		using RayAuxData					= RayAuxD;
		static const auto AuxInitFunc		= AuxFunc;

	private:
		// Options
		HitOpts								optsHit;
		ShadeOpts							optsShade;
		const TracerOptions					options;
		//
		const RayAuxData					initalValues;
		// Mappings for Kernel Calls (A.K.A. Batches)
		const GPUBaseAcceleratorI&			baseAccelerator;
		const AcceleratorBatchMappings&		accelerators;
		const MaterialBatchMappings&		materials;

	protected:
	public:
		// Constructors & Destructor
											TracerBaseLogic(const GPUBaseAcceleratorI&	baseAccelerator,
															const AcceleratorBatchMappings&,
															const MaterialBatchMappings&,
															const TracerOptions& options);
		virtual								~TracerBaseLogic() = default;

		// Init & Load
		TracerError							Initialize() override;
		SceneError							LoadScene(const std::string&) override;

		// Generate Camera Rays
		void								GenerateCameraRays(RayMemory&, RNGMemory&,
															   const CameraPerspective& camera,
															   const uint32_t samplePerPixel,
															   const Vector2ui& resolution,
															   const Vector2ui& pixelStart,
															   const Vector2ui& pixelCount) override;

		// Interface fetching for logic
		const GPUBaseAcceleratorI&			BaseAcelerator() override { return baseAccelerator; }
		const AcceleratorBatchMappings&		AcceleratorBatches() override { return accelerators; }
		const MaterialBatchMappings&		MaterialBatches() override { return materials; }

		// Returns bitrange of keys (should complement each other to 32-bit)
		const Vector2i&					   MaterialBitRange() const override { return options.materialKeyRange; }
		const Vector2i&					   AcceleratorBitRange() const override { return options.acceleratorKeyRange; }

		// Options of the Hitman & Shademan
		const HitOpts&					   HitOptions() const override { return optsHit; }
		const ShadeOpts&				   ShadeOptions() const override { return optsShade; }

		// Loads/Unloads material to GPU Memory
		void							   LoadMaterial(int gpuId, uint32_t matId) override;
		void							   UnloadMaterial(int gpuId, uint32_t matId) override;

		// Generates/Removes accelerator
		void							   GenerateAccelerators() override;

		// Misc
		// Retuns "sizeof(RayAux)"
		size_t							   PerRayAuxDataSize() const override { return sizeof(RayAux); }
		size_t							   HitStructSize() const override { return options.hitStructMaxSize; };
};
