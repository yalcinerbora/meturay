#pragma once

#include <set>
#include <memory>

#include "RayLib/SceneNodeI.h"

using SceneNodePtr = std::unique_ptr<SceneNodeI>;

struct SceneNodePtrLess
{
    bool operator()(const SceneNodePtr& a,
                    const SceneNodePtr& b)
    {
        return ((*a) < (*b));
    }
};

using NodeListing = std::set<std::unique_ptr<SceneNodeI>, SceneNodePtrLess>;