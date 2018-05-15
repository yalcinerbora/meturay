#pragma once
/**



*/

#include "SceneAcceleratorI.h"
#include "BVHDevice.cuh"

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
