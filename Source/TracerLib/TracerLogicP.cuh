#pragma once

#include "TracerLogicI.h"
#include "GPUPrimitiveI.h"

struct TracerError;

template<class RayAuxData>
using AuxInitFunc = void(*)(RayAuxData*,
							const uint32_t writeLoc,
							// Input
							const RayAuxData,
							// Index
							const Vector2ui& globalPixelId,
							const Vector2ui& localSampleId,
							const uint32_t samplePerPixel);

template<class RayAuxD, AuxInitFunc<RayAuxD> AuxF>
class TracerBaseLogic : public TracerBaseLogicI
{
	public:
		using RayAuxData					= RayAuxD;
		static constexpr auto AuxFunc		= AuxF;

	private:
	protected:
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

	public:
		// Constructors & Destructor
											TracerBaseLogic(const GPUBaseAcceleratorI& baseAccelerator,
															const AcceleratorBatchMappings&,
															const MaterialBatchMappings&,
															const TracerOptions& options,
															const RayAuxData& initalRayAux);
		virtual								~TracerBaseLogic() = default;

		// Interface
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
		const Vector2i					   SceneMaterialMaxBits() const override;
		const Vector2i					   SceneAcceleratorMaxBits() const override;

		// Options of the Hitman & Shademan
		const HitOpts&					   HitOptions() const override { return optsHit; }
		const ShadeOpts&				   ShadeOptions() const override { return optsShade; }

		// Misc
		// Retuns "sizeof(RayAux)"
		size_t							   PerRayAuxDataSize() const override { return sizeof(RayAuxData); }
		// Return mimimum size of an arbitrary struct which holds all hit results
		size_t							   HitStructSize() const override { return options.hitStructMaxSize; };
};

template<class RayAuxD, AuxInitFunc<RayAuxD> AuxFunc>
TracerBaseLogic<RayAuxD, AuxFunc>::TracerBaseLogic(const GPUBaseAcceleratorI& baseAccelerator,
												   const AcceleratorBatchMappings& a,
												   const MaterialBatchMappings& m,
												   const TracerOptions& options,
												   const RayAuxData& initialValues)
	: baseAccelerator(baseAccelerator)
	, accelerators(a)
	, materials(m)
	, options(options)
	, initalValues(initalValues)
{}

template<class RayAuxD, AuxInitFunc<RayAuxD> AuxFunc>
const Vector2i TracerBaseLogic<RayAuxD, AuxFunc>::SceneMaterialMaxBits() const
{
	//
	return Zero2i;
}

template<class RayAuxD, AuxInitFunc<RayAuxD> AuxFunc>
const Vector2i TracerBaseLogic<RayAuxD, AuxFunc>::SceneAcceleratorMaxBits() const
{
	//
	return Zero2i;
}

template<class RayAuxD, AuxInitFunc<RayAuxD> AuxFunc>
void TracerBaseLogic<RayAuxD, AuxFunc>::GenerateCameraRays(RayMemory&, RNGMemory&,
														   const CameraPerspective& camera,
														   const uint32_t samplePerPixel,
														   const Vector2ui& resolution,
														   const Vector2ui& pixelStart,
														   const Vector2ui& pixelCount)
{
	// 
}