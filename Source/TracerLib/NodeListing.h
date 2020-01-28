#pragma once

#include <set>
#include <memory>

#include "RayLib/SceneNodeI.h"

struct SceneNodePtrLess
{
    bool operator()(const SceneNodePtr& a,
                    const SceneNodePtr& b) const
    {
        return ((*a) < (*b));
    }
};

using NodeListing = std::set<SceneNodePtr, SceneNodePtrLess>;