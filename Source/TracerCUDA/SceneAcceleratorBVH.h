#pragma once
/**



*/

#include "SceneAcceleratorI.h"
#include "RayLib/BVHDevice.cuh"

class SceneAcceleratorBVH : SceneAcceleratorI
{		
	private:
		BVHDevice			boundingVolume;

	protected:
	public:
		// Constructors & Destructor
							SceneAcceleratorBVH();

							~SceneAcceleratorBVH();






};
