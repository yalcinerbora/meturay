#pragma once

#include <cstdint>

class BVH
{
	private:
		struct alignas(16) BVHNode
		{
			uint32_t left;
			uint32_t right;
			uint32_t parent;
		};

	protected:
	public:


};
