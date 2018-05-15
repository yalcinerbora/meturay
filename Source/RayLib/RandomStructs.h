#pragma once

#include <cstdint>

struct ConstRandomStackGMem
{
	const uint32_t* state;	
};

struct RandomStackGMem
{
	uint32_t* state;

	constexpr operator ConstRandomStackGMem() const;
};

constexpr RandomStackGMem::operator ConstRandomStackGMem() const
{
	return {state};
}
