#pragma once

#include <list>

#include "RayLib/SceneI.h"
#include "VolumeGPU.cuh"
#include "MaterialGPU.cuh"

struct SceneFile;

using LightList = std::vector<std::reference_wrapper<LightI>>;

class SceneGPU : SceneI
{
	private:
		MaterialList					materials;
		SurfaceList						surfaces;
		AnimatableList					animatables;
		LightList						lights;
		VolumeList						volumes;
		MeshBatchList					batches;

		// Volumes
		std::list<NCVolumeGPU>			ncVolumes;

		// Materials
		std::list<FluidMaterialGPU>		fMaterials;

		// GPU Data		

		DeviceMemory					memory;
		Vector2ui*						d_surfaceTypeIndexList;
		//MeshGPUData*					d_Meshes;
		VolumeDeviceData*				d_Volumes;

	protected:
	public:
		// Constructors & Destructor
										SceneGPU() = default;
										SceneGPU(const SceneFile& scene);
										SceneGPU(const SceneGPU&) = delete;
										SceneGPU(SceneGPU&&) = default;										
		SceneGPU&						operator=(const SceneGPU&) = delete;
		SceneGPU&						operator=(SceneGPU&&) = default;
										~SceneGPU() = default;

		// Material(Shading) Related	
		const MaterialI&				Material(uint32_t id) const override;
		const MaterialList&				Materials() const override;

		// Hit Related
		const SurfaceI&					Surface(uint32_t id) const override;
		const SurfaceList&				Surfaces() const override;

		// Animation Related
		AnimatableList&					Animatables() override;

		// Mesh - Volume Related
		const MeshBatchI&				MeshBatch(uint32_t id) override;
		const MeshBatchList&			MeshBatch() override;

		VolumeI&						Volume(uint32_t id) override;
		VolumeList&						Volumes() override;

		// Lights
		const LightI&					Light(uint32_t id) override;

		// Misc
		void							ChangeTime(double timeSec) override;

		//
		void							HitRays(uint32_t* location,
												const ConstRayRecordGMem,
												uint32_t rayCount) const override;
};