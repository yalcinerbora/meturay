#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <chrono>
using namespace std::chrono_literals;

#include "TracerCUDA/TracerCUDAEntry.h"
#include "RayLib/TracerLogicI.h"
#include "RayLib/GPUAcceleratorI.h"
#include "RayLib/GPUMaterialI.h"

#include "RayLib/Random.cuh"
#include "RayLib/RNGMemory.h"
#include "RayLib/RayMemory.h"

#include "RayLib/CameraKernels.cuh"

struct RayAuxGMem {};
struct RayAuxBaseData{};

//template <class RayAuxGMem, class RayAuxBaseData>
__device__ void AuxInitEmpty(const RayAuxGMem,
							 const uint32_t writeLoc,
							 // Input
							 const RayAuxBaseData,
							 // Index
							 const Vector2ui& globalPixelId,
							 const Vector2ui& localSampleId,
							 const uint32_t samplePerPixel)
{}

class MockTracerLogic : public TracerLogicI
{
	public:
		class BaseAcceleratorMock : public GPUBaseAcceleratorI
		{
			private:
				RNGMemory& 		rngMemory;

			public:
				// Constructors & Destructor
								BaseAcceleratorMock(RNGMemory& r) : rngMemory(r) {}
								~BaseAcceleratorMock() = default;

				void			Hit(// Output
									HitKey* dKeys,
									// Inputs
									const RayGMem* dRays,
									const RayId* dRayIds,
									uint32_t rayCount) override;
		};

		class AcceleratorMock : public GPUAcceleratorI
		{
			private:
				RNGMemory& 		rngMemory;

			public:
				// Constructors & Destructor
								AcceleratorMock(RNGMemory& r) : rngMemory(r) {}
								~AcceleratorMock() = default;


				void			Hit(// Output
									HitGMem* dHits,
									// Inputs
									const RayGMem* dRays,
									const RayId* dRayIds,
									uint32_t rayCount) override;
		};

		class MaterialMock : public GPUMaterialI
		{
			private:
				RNGMemory& 		rngMemory;


			public:
				// Constructors & Destructor
								MaterialMock(RNGMemory& r) : rngMemory(r) {}
								~MaterialMock() = default;

				void			ShadeRays(RayGMem* dRayOut,
										  void* dRayAuxOut,
										  //  Input
										  const RayGMem* dRayIn,
										  const HitGMem* dHitId,
										  const void* dRayAuxIn,
										  const RayId* dRayId,

										  const uint32_t rayCount,
										  RNGMemory& rngMem) override;

				uint8_t			MaxOutRayPerRay() const override { return 2; }
		};

	private:
		RNGMemory									rngMemory;

		HitOpts										optsHit;
		ShadeOpts									optsShade;

		static constexpr Vector2i					MaterialRange = Vector2i(0, 24);
		static constexpr Vector2i					AcceleratorRange = Vector2i(24, 32);
		
		static const std::string					HitName;
		static const std::string					ShadeName;

		// Mock Implementations
		BaseAcceleratorMock							baseAccelerator;
		std::map<uint16_t, GPUAcceleratorI*>		accelerators;
		const std::map<uint32_t, GPUMaterialI*>		materials;

	protected:
	public:
		// Constructors & Destructor
													MockTracerLogic(uint32_t seed);
		virtual										~MockTracerLogic() = default;


		// Generate Camera Rays
		void										GenerateCameraRays(RayMemory&, RNGMemory&,
																	   const CameraPerspective& camera,
																	   const uint32_t samplePerPixel,
																	   const Vector2ui& resolution,
																	   const Vector2ui& pixelStart,
																	   const Vector2ui& pixelCount) override;

		// Accessors for Managers
		// Hitman is responsible for
		const std::string&								HitmanName() const override { return HitName; }
		const std::string&								ShademanName() const override { return ShadeName; }

		// Interface fetching for logic
		GPUBaseAcceleratorI*							BaseAcelerator() override { return &baseAccelerator; }
		const std::map<uint16_t, GPUAcceleratorI*>&		Accelerators() override { return accelerators; }
		const std::map<uint32_t, GPUMaterialI*>&		Materials() override { return materials; }

		// Returns bitrange of keys (should complement each other to 32-bit)
		const Vector2i&									MaterialBitRange() const override { return MaterialRange; }
		const Vector2i&									AcceleratorBitRange() const override { return AcceleratorRange; }

		// Options of the Hitman & Shademan
		const HitOpts&									HitOptions() const override { return optsHit; }
		const ShadeOpts&								ShadeOptions() const override { return optsShade; }

		// Loads/Unloads material to GPU Memory
		void											LoadMaterial(int gpuId, uint32_t matId) override {}
		void											UnloadMaterial(int gpuId, uint32_t matId) override {}

		// Generates/Removes accelerator
		void											GenerateAccelerator(uint32_t accId) override {}
		void											RemoveAccelerator(uint32_t accId) override {}

		// Misc
		// Retuns "sizeof(RayAux)"
		size_t											PerRayAuxDataSize() override { return 0; }
};


const std::string MockTracerLogic::HitName = "";
const std::string MockTracerLogic::ShadeName = "";

void MockTracerLogic::BaseAcceleratorMock::Hit(// Output
											   HitKey* dKeys,
											   // Inputs
											   const RayGMem* dRays,
											   const RayId* dRayIds,
											   uint32_t rayCount)
{

}

void MockTracerLogic::AcceleratorMock::Hit(// Output
										   HitGMem* dHits,
										   // Inputs
										   const RayGMem* dRays,
										   const RayId* dRayIds,
										   uint32_t rayCount)
{

}

void MockTracerLogic::MaterialMock::ShadeRays(RayGMem* dRayOut,
											  void* dRayAuxOut,
											  //  Input
											  const RayGMem* dRayIn,
											  const HitGMem* dHitId,
											  const void* dRayAuxIn,
											  const RayId* dRayId,

											  const uint32_t rayCount,
											  RNGMemory& rngMem)
{

}

void MockTracerLogic::GenerateCameraRays(RayMemory& rMem,
										 RNGMemory& rngMemory,
										 const CameraPerspective& camera,
										 const uint32_t samplePerPixel,
										 const Vector2ui& resolution,
										 const Vector2ui& pixelStart,
										 const Vector2ui& pixelCount)
{
	RayAuxGMem rAux;
	RayAuxBaseData rAuxBase;

	// Camera Ray Generation Kernel Check
	KCGenerateCameraRays<RayAuxGMem, RayAuxBaseData, AuxInitEmpty><<<1, 1>>>
	(
		rMem.RaysOut(),
		rAux,
		// Input
		rngMemory.RNGData(0),
		camera,
		samplePerPixel,
		resolution,
		pixelStart,
		pixelCount,
		//
		rAuxBase
	);
	CUDA_KERNEL_CHECK();
}

MockTracerLogic::MockTracerLogic(uint32_t seed)
	: rngMemory(seed)
	, baseAccelerator(rngMemory)
{

	// Generate Fake stuff
	// 


}

TEST(CameraRayGPU, Test)
{
	constexpr Vector2ui resolution = Vector2ui(100, 100);
	constexpr uint32_t seed = 0;

	// Create our mock
	MockTracerLogic mockLogic(seed);

	// Load Tracer DLL
	auto tracerI = CreateTracerCUDA();
	tracerI->Initialize(seed, &mockLogic);
	tracerI->ResizeImage(resolution);
	tracerI->ReportionImage();

	// Generate Camera Rays
	// Mock Tracer only cares about pixel count and sample size
	tracerI->GenerateCameraRays(CameraPerspective{}, 1);
	// Loop until all rays are processed
	while(tracerI->Continue())
	{
		tracerI->Render();
	}
	tracerI->FinishSamples();

	// All Done!!
}