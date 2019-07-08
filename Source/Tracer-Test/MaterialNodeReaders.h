#pragma once

#include <set>

struct ConstantAlbedoMatData;
struct ConstantBoundaryMatData;
class SceneNodeI;
class DeviceMemory;

extern ConstantAlbedoMatData ConstantAlbedoMatRead(DeviceMemory&,
                                                   const std::set<SceneNodeI>& materialNodes,
                                                   double);

extern ConstantBoundaryMatData ConstantBoundaryMatRead(const std::set<SceneNodeI>& materialNodes,
                                                       double);