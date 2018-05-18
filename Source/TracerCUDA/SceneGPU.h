#pragma once

#include "RayLib/SceneI.h"

using LightList = std::vector<std::reference_wrapper<LightI>>;

class SceneGPU : SceneI
{
	private:
	MaterialList		materials;
	SurfaceList			surfaces;
	AnimatableList		animatables;
	LightList			lights;
	VolumeList			volumes;
	MeshBatchList		batches;


	protected:
	public:
		// Constructors & Destructor
										SceneGPU();
										SceneGPU(const std::string& fileName);


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