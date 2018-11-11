#pragma once

#include "TracerLogicI.h"
#include "GPUPrimitiveI.h"
#include "AuxiliaryDataKernels.cuh"
#include "CameraKernels.cuh"

struct TracerError;

template<class RayAuxD, AuxInitFunc<RayAuxD> AuxF,
		 RayFinalizeFunc<RayAuxD> FinalizeF>
class TracerBaseLogic : public TracerBaseLogicI
{
	public:
		using RayAuxData					= RayAuxD;
		static constexpr auto AuxFunc		= AuxF;
		static constexpr auto FinalizeFunc	= FinalizeF;

	private:
	protected:
		// Options
		HitOpts								optsHit;
		ShadeOpts							optsShade;
		const TracerOptions					options;
		//
		const RayAuxData					initalValues;
		// Mappings for Kernel Calls (A.K.A. Batches)
		GPUBaseAcceleratorI&				baseAccelerator;
		const AcceleratorBatchMappings&		accelerators;
		const MaterialBatchMappings&		materials;

	public:
		// Constructors & Destructor
											TracerBaseLogic(GPUBaseAcceleratorI& baseAccelerator,
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
		GPUBaseAcceleratorI&				BaseAcelerator() override { return baseAccelerator; }
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

template<class RayAuxD,
		 AuxInitFunc<RayAuxD> AuxF,
		 RayFinalizeFunc<RayAuxD> FinalizeF>
TracerBaseLogic<RayAuxD, AuxF, FinalizeF>::TracerBaseLogic(GPUBaseAcceleratorI& baseAccelerator,
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

template<class RayAuxD,
		 AuxInitFunc<RayAuxD> AuxF,
		 RayFinalizeFunc<RayAuxD> FinalizeF>
const Vector2i TracerBaseLogic<RayAuxD, AuxF, FinalizeF>::SceneMaterialMaxBits() const
{
	//
	return Zero2i;
}

template<class RayAuxD,
		 AuxInitFunc<RayAuxD> AuxF,
		 RayFinalizeFunc<RayAuxD> FinalizeF>
const Vector2i TracerBaseLogic<RayAuxD, AuxF, FinalizeF>::SceneAcceleratorMaxBits() const
{
	//
	return Zero2i;
}

template<class RayAuxD,
	AuxInitFunc<RayAuxD> AuxF,
	RayFinalizeFunc<RayAuxD> FinalizeF>
	void TracerBaseLogic<RayAuxD, AuxF, FinalizeF>::GenerateCameraRays(RayMemory&, RNGMemory&,
																	   const CameraPerspective& camera,
																	   const uint32_t samplePerPixel,
																	   const Vector2ui& resolution,
																	   const Vector2ui& pixelStart,
																	   const Vector2ui& pixelCount)
{
	// 
}