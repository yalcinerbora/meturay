#pragma once

#include <set>
#include <memory>

#include "RayLib/SceneNodeI.h"

using SceneNodePtr = std::unique_ptr<SceneNodeI>;

static const auto SceneNodePtrLess = [](const SceneNodePtr& a,
                                        const SceneNodePtr& b) -> bool
{
    return ((*a) < (*b));
};

using NodeListing = std::set<std::unique_ptr<SceneNodeI>, decltype(SceneNodePtrLess)>;
