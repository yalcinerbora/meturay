#pragma once

#include <json.hpp>

struct SceneFileNode
{
	uint32_t id;
	nlohmann::json jsn;

	bool operator<(const SceneFileNode&) const;
};

inline bool SceneFileNode::operator<(const SceneFileNode& node) const
{
	return id < node.id;
}