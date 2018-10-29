#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <random>
#include <chrono>
using namespace std::chrono_literals;

#include "RayLib/Log.h"
#include "RayLib/TracerError.h"

#include "TracerLib/TracerBase.h"
#include "TracerLib/TracerLogicI.h"
#include "TracerLib/GPUAcceleratorI.h"
#include "TracerLib/GPUMaterialI.h"

#include "TracerLib/Random.cuh"
#include "TracerLib/RNGMemory.h"
#include "TracerLib/RayMemory.h"

#include "TracerLib/CameraKernels.cuh"

#include "TracerLib/TracerLoader.h"

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

class MockTracerLogic : public TracerBaseLogicI
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
				
				// Type(as string) of the accelerator group
				const char*			Type() const override { return "MockBaseAccel"; }
				// KC
				void				Hit(// Output
										TransformId* dTransformIds,
										HitKey* dAcceleratorKeys,
										// Inputs
										const RayGMem* dRays,
										const RayId* dRayIds,
										const uint32_t rayCount) const override;

				void			Constrcut(// List of allocator hitkeys of surfaces
										  const std::map<uint32_t, HitKey>&,
										  // List of all Surface/Transform pairs
										  // that will be constructed
										  const std::map<uint32_t, uint32_t>&) override {};
				void			Reconstruct(// List of allocator hitkeys of surfaces
											const std::map<uint32_t, HitKey>&,
											// List of changed Surface/Transform pairs
											const std::map<uint32_t, uint32_t>&) override {};
		};

		class AcceleratorMock : public GPUAcceleratorBatchI
		{
			private:
				MockTracerLogic& 				mockLogic;
				uint32_t						myKey;
				const void*						groupNullPtr = nullptr;

			public:
				// Constructors & Destructor
												AcceleratorMock(MockTracerLogic& r, uint32_t myKey) 
													: mockLogic(r), myKey(myKey)  {}
												~AcceleratorMock() = default;

				// Type(as string) of the accelerator group
				const char*						Type() const override { return "MockAccelBatch"; }
				// KC
				void							Hit(// O
													HitKey* dMaterialKeys,
													PrimitiveId* dPrimitiveIds,
													HitStructPtr dHitStructs,
													// I-O													
													RayGMem* dRays,
													// Input
													const TransformId* dTransformIds,
													const RayId* dRayIds,
													const HitKey* dAcceleratorKeys,
													const uint32_t rayCount) const override;

				const GPUPrimitiveGroupI&		PrimitiveGroup() const override;
				const GPUAcceleratorGroupI&		AcceleratorGroup() const override;
		};

		class MaterialMock : public GPUMaterialBatchI
		{
			private:
				MockTracerLogic& 	mockLogic;
				bool				isMissMaterial;
				const void*			groupNullPtr = nullptr;

			public:
				// Constructors & Destructor
											MaterialMock(MockTracerLogic& r, bool missMat) 
												: mockLogic(r)
												, isMissMaterial(missMat) {}
											~MaterialMock() = default;

				// Type(as string) of the accelerator group
				const char*					Type() const override { return "MockMatBatch"; }
				// KC
				void						ShadeRays(// Output
													  RayGMem* dRayOut,
													  void* dRayAuxOut,
													  //  Input
													  const RayGMem* dRayIn,
													  const void* dRayAuxIn,
													  const PrimitiveId* dPrimitiveIds,
													  const HitStructPtr dHitStructs,
													  //
													  const HitKey* dMatIds,
													  const RayId* dRayIds,

													  const uint32_t rayCount,
													  RNGMemory& rngMem) const override;

				// Every MaterialBatch is available for a specific primitive / material data
				const GPUPrimitiveGroupI&	PrimitiveGroup() const override;
				const GPUMaterialGroupI&	MaterialGroup() const override;

				uint8_t						OutRayCount() const override { return isMissMaterial ? 0 : 1; }
		};

	private:
		std::mt19937								rng;	
		std::uniform_int_distribution<>				matIndexGenerator;

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

		AcceleratorBatchMappings					accelerators;
		MaterialBatchMappings						materials;
		
		// Convenience
		std::vector<HitKey>							materialKeys;
		std::vector<HitKey>							acceleratorKeys;
			   
		static constexpr int						AcceleratorCount = 2;
		static constexpr int						MaterialCount = 4;

	protected:
	public:
		// Constructors & Destructor
													MockTracerLogic(uint32_t seed);
		virtual										~MockTracerLogic() = default;

		// Init & Load
		TracerError									Initialize() override;
	
		// Generate Camera Rays
		void										GenerateCameraRays(RayMemory&, RNGMemory&,
																	   const CameraPerspective& camera,
																	   const uint32_t samplePerPixel,
																	   const Vector2ui& resolution,
																	   const Vector2ui& pixelStart,
																	   const Vector2ui& pixelCount) override;
		void										GenerateRays(RayMemory&, RNGMemory&,
																 const uint32_t rayCount) override;

		

		// Interface fetching for logic
		const GPUBaseAcceleratorI&					BaseAcelerator() override { return *baseAccelerator; }
		const AcceleratorBatchMappings&				AcceleratorBatches() override { return accelerators; }
		const MaterialBatchMappings&				MaterialBatches() override { return materials; }

		// Returns max bits of keys (for batch and id respectively)
		const Vector2i								SceneMaterialMaxBits() const override { return MaterialRange; }
		const Vector2i								SceneAcceleratorMaxBits() const override { return AcceleratorRange; }

		// Options of the Hitman & Shademan
		const HitOpts&								HitOptions() const override { return optsHit; }
		const ShadeOpts&							ShadeOptions() const override { return optsShade; }

		// Misc
		// Retuns "sizeof(RayAux)"
		size_t										PerRayAuxDataSize() const override { return 0; }
		// Return mimimum size of an arbitrary struct which holds all hit results
		size_t										HitStructSize() const override { return sizeof(uint32_t); };
};

const GPUPrimitiveGroupI& MockTracerLogic::AcceleratorMock::PrimitiveGroup() const 
{ 
	return *static_cast<const GPUPrimitiveGroupI*>(groupNullPtr);
}

const GPUAcceleratorGroupI& MockTracerLogic::AcceleratorMock::AcceleratorGroup() const
{
	return *static_cast<const GPUAcceleratorGroupI*>(groupNullPtr);
}

const GPUPrimitiveGroupI& MockTracerLogic::MaterialMock::PrimitiveGroup() const
{
	return *static_cast<const GPUPrimitiveGroupI*>(groupNullPtr);
}

const GPUMaterialGroupI& MockTracerLogic::MaterialMock::MaterialGroup() const
{
	return *static_cast<const GPUMaterialGroupI*>(groupNullPtr);
}

const std::string MockTracerLogic::HitName = "";
const std::string MockTracerLogic::ShadeName = "";

void MockTracerLogic::BaseAcceleratorMock::Hit(// Output
											   TransformId* dTransformIds,
											   HitKey* dAcceleratorKeys,
											   // Inputs
											   const RayGMem* dRays,
											   const RayId* dRayIds,
											   const uint32_t rayCount) const
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
			dAcceleratorKeys[i] = HitKey::OutsideMatKey;
		else
			dAcceleratorKeys[i] = mockLogic.acceleratorKeys[index - 1];
	}
}

void MockTracerLogic::AcceleratorMock::Hit(// O
										   HitKey* dMaterialKeys,
										   PrimitiveId* dPrimitiveIds,
										   HitStructPtr dHitStructs,
										   // I-O													
										   RayGMem* dRays,
										   // Input
										   const TransformId* dTransformIds,
										   const RayId* dRayIds,
										   const HitKey* dAcceleratorKeys,
										   const uint32_t rayCount) const
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
			// We found a hit,
			// Randomly select a material for hit			
			HitKey materialId = mockLogic.materialKeys[mockLogic.matIndexGenerator(mockLogic.rng)];
			dMaterialKeys[rayId] = materialId;
			// Put primitive id
			dPrimitiveIds[rayId] = 0;
			// Put a struct
			struct Test
			{
				int i;
			};
			//Test a;
			//dHitStructs[static_cast<int>(i)] = a;
			
		}
	}
	printf("\n");
}

void MockTracerLogic::MaterialMock::ShadeRays(// Output
											  RayGMem* dRayOut,
											  void* dRayAuxOut,
											  //  Input
											  const RayGMem* dRayIn,
											  const void* dRayAuxIn,
											  const PrimitiveId* dPrimitiveIds,
											  const HitStructPtr dHitStructs,
											  //
											  const HitKey* dMatIds,
											  const RayId* dRayIds,

											  const uint32_t rayCount,
											  RNGMemory& rngMem) const
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
	materials.emplace(std::make_pair(static_cast<uint32_t>(HitKey::OutsideMatKey),
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

void MockTracerLogic::GenerateRays(RayMemory&, RNGMemory&,
								   const uint32_t rayCount)
{}

MockTracerLogic::MockTracerLogic(uint32_t seed)
	: seed(seed)
	, matIndexGenerator(0, MaterialCount - 1)
{}

TEST(MockTracerTest, Test)
{
	constexpr Vector2ui resolution = Vector2ui(3, 3);
	constexpr uint32_t seed = 0;

	// Create our mock
	MockTracerLogic mockLogic(seed);

	// Load Tracer DLL
	TracerBase tracer;
	TracerI* tracerI = &tracer;
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