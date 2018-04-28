#pragma once
/**

Sub-space of the array with a unique id

*/

template<class T>
struct ArrayPortion
{
	T		portionId;
	size_t	offset;
	size_t	count;
};
