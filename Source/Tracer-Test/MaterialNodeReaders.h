#pragma once

#include <set>

struct ConstantAlbedoMatData;
struct ConstantBoundaryMatData;
struct SceneFileNode;

class DeviceMemory;

extern ConstantAlbedoMatData ConstantAlbedoMatRead(DeviceMemory&,
												   const std::set<SceneFileNode>& materialNodes,
												   double);

extern ConstantBoundaryMatData ConstantBoundaryMatRead(const std::set<SceneFileNode>& materialNodes,
													   double);