#pragma once

#include "TracerLogicI.h"
#include "GPUScene.h"

struct TracerError;

//template<class TracerLogic>
//class TracerLogicGenerator : public TracerLogicGeneratorI
//{
//	// Logic Generators
//	virtual SceneError							GetPrimitiveGroup(GPUPrimitiveGroupI*&,
//																  const std::string& primitiveType) = 0;
//	virtual SceneError							GetAcceleratorGroup(GPUAcceleratorGroupI*&,
//																	const GPUPrimitiveGroupI&,
//																	const std::string& accelType) = 0;
//	virtual SceneError							GetMaterialGroup(GPUMaterialGroupI*&,
//																 const GPUPrimitiveGroupI&,
//																 const std::string& materialType) = 0;
//};

template<class RayAuxType, class RayAuxInit>
class TracerLogic : public TracerLogicI
{
	public:
		using RayAux								= RayAuxType;
	
		static __device__ void						AuxInitEmpty(const RayAuxGMem*,
																 const uint32_t writeLoc,
																 // Input
																 const RayAuxInit,
																 // Index
																 const Vector2ui& globalPixelId,
																 const Vector2ui& localSampleId,
																 const uint32_t samplePerPixel);

	private:
		// Abstract Factory for types
		TracerLogicGeneratorI&						typeGenerator;
		GPUScene									scene;
		// Options
		HitOpts										optsHit;
		ShadeOpts									optsShade;
		const TracerOptions							options;
		//
		RayAuxInit									auxInitStruct;
		// Mappings for Kernel Call
		GPUBaseAcceleratorI*						baseAccelerator;
		AcceleratorBatchMappings					accelerators;
		MaterialBatchMappings						materials;


	protected:
	public:
		// Constructors & Destructor
													TracerLogic(const TracerOptions& options);
		virtual										~TracerLogic() = default;

		// Init & Load
		TracerError									Initialize() override;
		SceneError									LoadScene(const std::string&) override;

		// Generate Camera Rays
		void										GenerateCameraRays(RayMemory&, RNGMemory&,
																	   const CameraPerspective& camera,
																	   const uint32_t samplePerPixel,
																	   const Vector2ui& resolution,
																	   const Vector2ui& pixelStart,
																	   const Vector2ui& pixelCount) override;


		// Interface fetching for logic
		GPUBaseAcceleratorI*						BaseAcelerator() override { return baseAccelerator; }
		const AcceleratorBatchMappings&				AcceleratorBatches() override { return accelerators; }
		const MaterialBatchMappings&				MaterialBatches() override { return materials; }

		// Returns bitrange of keys (should complement each other to 32-bit)
		const Vector2i&								MaterialBitRange() const override { return options.materialKeyRange; }
		const Vector2i&								AcceleratorBitRange() const override { return options.acceleratorKeyRange; }

		// Options of the Hitman & Shademan
		const HitOpts&								HitOptions() const override { return optsHit; }
		const ShadeOpts&							ShadeOptions() const override { return optsShade; }

		// Loads/Unloads material to GPU Memory
		void										LoadMaterial(int gpuId, uint32_t matId) override;
		void										UnloadMaterial(int gpuId, uint32_t matId) override;

		// Generates/Removes accelerator
		void										GenerateAccelerators() override;

		// Misc
		// Retuns "sizeof(RayAux)"
		size_t										PerRayAuxDataSize() const override { return sizeof(RayAux); }
		size_t										HitStructSize() const override { return typeGenerator.CurrentMinHitSize(); };
};
