#pragma once

#include <set>

struct ColorMaterialData;
struct SceneFileNode;
class DeviceMemory;

extern ColorMaterialData ColorMaterialRead(DeviceMemory&,
										   const std::set<SceneFileNode>& materialNodes,
										   double);