#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>
#include <chrono>
using namespace std::chrono_literals;

#include "TracerCUDA/TracerCUDAEntry.h"
#include "RayLib/TracerLogicI.h"
#include "RayLib/GPUAcceleratorI.h"
#include "RayLib/GPUMaterialI.h"

#include "RayLib/Random.cuh"
#include "RayLib/RNGMemory.h"
#include "RayLib/RayMemory.h"
#include "RayLib/Log.h"

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
				MockTracerLogic& 	mockLogic;

			public:
				// Constructors & Destructor
									BaseAcceleratorMock(MockTracerLogic& r) : mockLogic(r) {}
									~BaseAcceleratorMock() = default;

				void				Hit(// Output
										HitKey* dKeys,
										// Inputs
										const RayGMem* dRays,
										const RayId* dRayIds,
										uint32_t rayCount) override;
		};

		class AcceleratorMock : public GPUAcceleratorGroupI
		{
			private:
				MockTracerLogic& 	mockLogic;
				uint32_t			myKey;

			public:
				// Constructors & Destructor
									AcceleratorMock(MockTracerLogic& r, uint32_t myKey) 
										: mockLogic(r), myKey(myKey)  {}
									~AcceleratorMock() = default;


				void				Hit(// I-O
										RayGMem* dRays,
										void* dHitStructs,
										HitKey* dCurrentHits,
										// Input
										const RayId* dRayIds,
										const HitKey* dPotentialHits,
										const uint32_t rayCount)  override;
		};

		class MaterialMock : public GPUMaterialI
		{
			private:
				MockTracerLogic& 	mockLogic;
				bool				isMissMaterial;

			public:
				// Constructors & Destructor
									MaterialMock(MockTracerLogic& r, bool missMat) 
										: mockLogic(r)
										, isMissMaterial(missMat) {}
									~MaterialMock() = default;

				void				ShadeRays(RayGMem* dRayOut,
											  void* dRayAuxOut,
											  //  Input
											  const RayGMem* dRayIn,
											  const HitKey* dCurrentHits,
											  const void* dRayAuxIn,
											  const RayId* dRayIds,

											  const uint32_t rayCount,
											  RNGMemory& rngMem) override;

				uint8_t				MaxOutRayPerRay() const override { return isMissMaterial ? 0 : 1; }
		};

	private:
		std::mt19937								rng;	
		std::uniform_int_distribution<>				uniformRNGMaterial;
		std::uniform_int_distribution<>				uniformRNGAcceleratorl;

		uint32_t									seed;

		HitOpts										optsHit;
		ShadeOpts									optsShade;

		static constexpr Vector2i					MaterialRange = Vector2i(0, 24);
		static constexpr Vector2i					AcceleratorRange = Vector2i(24, 32);

		static const std::string					HitName;
		static const std::string					ShadeName;

		// Mock Implementations
		std::unique_ptr<BaseAcceleratorMock>		baseAccelerator;
		std::vector<AcceleratorMock>				mockAccelerators;
		std::vector<MaterialMock>					mockMaterials;

		std::map<uint16_t, GPUAcceleratorGroupI*>	accelerators;
		std::map<uint32_t, GPUMaterialI*>			materials;
		
		// Convenience
		std::vector<HitKey>							materialKeys;
			   
		static constexpr int						AcceleratorCount = 2;
		static constexpr int						MaterialCount = 4;

	protected:
	public:
		// Constructors & Destructor
													MockTracerLogic(uint32_t seed);
		virtual										~MockTracerLogic() = default;


		TracerError									Initialize() override;

		// Generate Camera Rays
		void										GenerateCameraRays(RayMemory&, RNGMemory&,
																	   const CameraPerspective& camera,
																	   const uint32_t samplePerPixel,
																	   const Vector2ui& resolution,
																	   const Vector2ui& pixelStart,
																	   const Vector2ui& pixelCount) override;

		// Accessors for Managers
		// Hitman is responsible for
		const std::string&									HitmanName() const override { return HitName; }
		const std::string&									ShademanName() const override { return ShadeName; }

		// Interface fetching for logic
		GPUBaseAcceleratorI*								BaseAcelerator() override { return &(*baseAccelerator); }
		const std::map<uint16_t, GPUAcceleratorGroupI*>&	AcceleratorGroups() override { return accelerators; }
		const std::map<uint32_t, GPUMaterialI*>&			Materials() override { return materials; }

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
		size_t											HitStructMaxSize() { return sizeof(uint32_t); };
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
	// Go To CPU
	CUDA_CHECK(cudaDeviceSynchronize());

	METU_LOG("-----------------------------");

	// Delegate Stuff Interleaved
	for(uint32_t i = 0; i < rayCount; i++)
	{
		// Key index is used to acces RayGMem (this program does not care about actual ray)
		//uint32_t keyIndex = dRayIds[i];

		// Each Iteration some of the rays are missed (only first ray in this case)
		uint32_t index = i % (AcceleratorCount * MaterialCount + 1);
		if(index == 0)
			dKeys[i] = HitConstants::OutsideMatKey;
		else
			dKeys[i] = mockLogic.materialKeys[index - 1];
	}
}

void MockTracerLogic::AcceleratorMock::Hit(// I-O
										   RayGMem* dRays,
										   void* dHitStructs,
										   HitKey* dCurrentHits,
										   // Input
										   const RayId* dRayIds,
										   const HitKey* dPotentialHits,
										   const uint32_t rayCount)
{
	// Go To CPU
	CUDA_CHECK(cudaDeviceSynchronize());


	// Each Individual Hit segment writes the actual hit result
	METU_LOG("Stub Accelerator Work %u", rayCount);
	std::stringstream s;	
	for(uint32_t i = 0; i < rayCount; i++)
	{
		RayId rayId = dRayIds[i];
		printf("%d, ", rayId);


		double random01 = static_cast<double>(mockLogic.rng()) /
						  static_cast<double>(mockLogic.rng.max());

		// %50 Make it hit
		if(random01 <= 0.5)
		{
			// Randomly select a material for hit			
			uint32_t materialId = static_cast<uint32_t>(mockLogic.uniformRNGMaterial(mockLogic.rng));
			uint32_t combinedIndex = myKey << MaterialRange[1];
			combinedIndex |= materialId;

			dCurrentHits[rayId] = combinedIndex;
		}
	}
	printf("\n");
}

void MockTracerLogic::MaterialMock::ShadeRays(RayGMem* dRayOut,
											  void* dRayAuxOut,
											  //  Input
											  const RayGMem* dRayIn,
											  const HitKey* dCurrentHits,
											  const void* dRayAuxIn,
											  const RayId* dRayIds,

											  const uint32_t rayCount,
											  RNGMemory& rngMem)
{
	// Go To CPU
	CUDA_CHECK(cudaDeviceSynchronize());

	METU_LOG("Stub Material Work %u", rayCount);
	for(uint32_t i = 0; i < rayCount; i++)
	{
		RayId rayId = dRayIds[i];
		printf("%d, ", rayId);
	}
	printf("\n");
}

TracerError MockTracerLogic::Initialize()
{
	// Initialize Single Here Also
	TracerError e(TracerError::END);
	if((e = CudaSystem::Initialize()) != TracerError::OK)
	{
		return e;
	}

	rng.seed(seed);
	baseAccelerator = std::make_unique<BaseAcceleratorMock>(*this);

	// Generate Accelerators and Id Mappings
	// Be careful pointers will be invalidated 
	// if vector reallocates. Thus pre-allocate for all
	// the data.
	mockAccelerators.reserve(AcceleratorCount);
	mockMaterials.reserve((MaterialCount * AcceleratorCount) + 1);
	for(int i = 0; i < AcceleratorCount; i++)
	{
		mockAccelerators.emplace_back(*this, static_cast<uint32_t>(i));
		accelerators.emplace(std::make_pair(static_cast<uint32_t>(i),
											&mockAccelerators.back()));

		for(int j = 0; j < MaterialCount; j++)
		{
			int combinedIndex = i << MaterialRange[1];
			combinedIndex |= j;

			mockMaterials.emplace_back(*this, false);
			materials.emplace(std::make_pair(static_cast<uint32_t>(combinedIndex),
											 &mockMaterials.back()));
			materialKeys.emplace_back(static_cast<uint32_t>(combinedIndex));
		}
	}

	// Create miss material
	mockMaterials.emplace_back(*this, true);
	materials.emplace(std::make_pair(static_cast<uint32_t>(HitConstants::OutsideMatKey),
									 &mockMaterials.back()));

	// We have total of 8 material seperated by 2 accelerators
	return TracerError::OK;
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
	constexpr int GPUId = 0;
	CudaSystem::GPUCallX(GPUId, rngMemory.SharedMemorySize(GPUId), 0,
						 KCGenerateCameraRays<RayAuxGMem, RayAuxBaseData, AuxInitEmpty>,
						 rMem.RaysOut(),
						 rAux,
						 // Input
						 rngMemory.RNGData(GPUId),
						 camera,
						 samplePerPixel,
						 resolution,
						 pixelStart,
						 pixelCount,
						 //
						 rAuxBase);
	// We do not use this actual data but w/e
}

MockTracerLogic::MockTracerLogic(uint32_t seed)
	: seed(seed)
	, uniformRNGMaterial(0, MaterialCount - 1)
	, uniformRNGAcceleratorl(0, MaterialCount - 1)
{}

TEST(MockTracerTest, Test)
{
	constexpr Vector2ui resolution = Vector2ui(3, 3);
	constexpr uint32_t seed = 0;

	// Create our mock
	MockTracerLogic mockLogic(seed);

	// Load Tracer DLL
	auto tracerI = CreateTracerCUDA();
	tracerI->Initialize(seed, mockLogic);
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