#pragma once

#include "RayLib/HitStructs.h"
#include "RayLib/Vector.h"
#include "RayLib/SceneStructs.h"

#include "Random.cuh"

template <class EstimatorData>
using EstimateEventFunc = void(*)(// Output
                                  HitKey& boundaryMatKey,
                                  Vector3& direction,
                                  float& probability,
                                  // Input
                                  const Vector3& position,
                                  RandomGPU& rng,
                                  //
                                  const EstimatorData&);

template <class EstimatorData>
using GenerateEstimator = void(*)(// Output
                                  EstimatorData&,
                                  // Input
                                  const HitKey* dBoundaryMatList);