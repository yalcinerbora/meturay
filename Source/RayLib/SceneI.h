#pragma once
/**

Scene Interface

Scene


*/

#include <vector>

class MaterialI;
class SurfaceI;
class AnimateI;

class MeshBatchI;
class VolumeI;
class MeshI;
class LightI;

using SurfaceList = std::vector<std::reference_wrapper<SurfaceI>>;
using MaterialList = std::vector<std::reference_wrapper<MaterialI>>;
using AnimatableList = std::vector<std::reference_wrapper<AnimateI>>;
using VolumeList = std::vector<std::reference_wrapper<VolumeI>>;
using MeshBatchList = std::vector<std::reference_wrapper<MeshBatchI>>;
using LightList = std::vector<std::reference_wrapper<LightI>>;

struct ConstRayRecordGMem;

class SceneI
{

	public:
		virtual								~SceneI() = default;

		// Material(Shading) Related
		virtual const MaterialI&			Material(uint32_t id) const = 0;
		virtual const MaterialList&			Materials() const = 0;

		// Hit Related
		virtual const SurfaceI&				Surface(uint32_t id) const = 0;
		virtual const SurfaceList&			Surfaces() const = 0;
	
		// Animation Related
		virtual AnimatableList&				Animatables() = 0;

		// Mesh - Volume Related
		virtual const MeshBatchI&			MeshBatch(uint32_t id) = 0;
		virtual const MeshBatchList&		MeshBatch() = 0;
	
		virtual VolumeI&					Volume(uint32_t id) = 0;
		virtual VolumeList&					Volumes() = 0;

		// Lights
		virtual const LightI&				Light(uint32_t id) = 0;

		// Misc
		virtual void						ChangeTime(double timeSec) = 0;

		//
		virtual	void						HitRays(uint32_t* location, 
													const ConstRayRecordGMem,
													uint32_t rayCount) const = 0;
};

