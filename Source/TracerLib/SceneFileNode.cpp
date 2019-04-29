#include "SceneFileNode.h"
#include "nlohmann/json.hpp"

SceneFileNode::SceneFileNode(const nlohmann::json& n)
    : node(new nlohmann::json(n))
    , id(0)
{}

SceneFileNode::SceneFileNode(const nlohmann::json& n, uint32_t id)
    : node(new nlohmann::json(n))
    , id(0)
{}

SceneFileNode::SceneFileNode(const SceneFileNode& other)
    : node (new nlohmann::json(*other.node))
    , id(other.id)
{}

SceneFileNode::SceneFileNode(SceneFileNode&& other)
    : node(other.node)
    , id(other.id)
{
    other.node = nullptr;
}

SceneFileNode::~SceneFileNode()
{
    if(node) delete node;
}