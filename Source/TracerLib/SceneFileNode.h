#pragma once
/**

Have to wrap json lib since it does not work properly
with nvcc

*/

#include <cstdint>
#include <cassert>
#include <nlohmann/json_fwd.hpp>

struct SceneFileNode
{
	private:
		const nlohmann::json*	node;
		uint32_t					id;

	protected:
	public:
		// Constructors & Destructor
							SceneFileNode(const nlohmann::json&);
							SceneFileNode(const nlohmann::json&, uint32_t id);
							
							SceneFileNode(const SceneFileNode&); 
							SceneFileNode(SceneFileNode&&);
		SceneFileNode&		operator=(const SceneFileNode&) = delete;
		SceneFileNode&		operator=(SceneFileNode&&);
							~SceneFileNode();

		uint32_t			Id() const;
							operator const nlohmann::json&() const;

		bool				operator<(const SceneFileNode& node) const;							
};

inline SceneFileNode::operator const nlohmann::json&() const
{
	assert(node);
	return *node;
}

inline uint32_t SceneFileNode::Id() const
{
	return id;
}

inline SceneFileNode& SceneFileNode::operator=(SceneFileNode&& other)
{
	assert(this != &other);
	node = other.node;
	other.node = nullptr;
	return *this;
}

inline bool SceneFileNode::operator<(const SceneFileNode& node) const
{
	return id < node.id;
}