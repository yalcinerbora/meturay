#pragma once

/**

State storing simple Thread data.
Can be set from outside of the tread
and shown

*/

#include <atomic>

template <class T>
class ThreadData
{
	private:
		std::atomic_bool		changedOutside;
		T						data;

	protected:
	public:
		ThreadData&				operator=(const T& t);
		bool					CheckChanged(T& newData);
};

template <class T>
inline ThreadData<T>& ThreadData<T>::operator=(const T& t)
{
	data = t;
	changedOutside = true;
	return *this;
}

template <class T>
inline bool ThreadData<T>::CheckChanged(T& newData)
{
	bool changed = changedOutside.exchange(false);
	newData = data;
	return changed;
}