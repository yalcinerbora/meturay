#pragma once

/**

Stated storage of simple Thread data.

Can be set from outside of the tread.
Thread will check the status that this variable is changed
and act accordingly.
*/

#include <atomic>
#include <mutex>

template <class T>
class ThreadVariable
{
    private:
        mutable std::mutex      mutex;
        bool                    changedOutside;
        T                       data;

    protected:
    public:
        // Constructors & Destructor
                                ThreadVariable();
                                ThreadVariable(const T&);
                                ThreadVariable(const ThreadVariable&) = delete;
                                ThreadVariable(ThreadVariable&&) = default;
        ThreadVariable&         operator=(const ThreadVariable&) = delete;
        ThreadVariable&         operator=(ThreadVariable&&) = default;
                                ~ThreadVariable() = default;

        // Additional Operators
        ThreadVariable&         operator=(const T& t);
        // Members
        bool                    CheckChanged(T& newData);
        T                       Get() const;
};

template <class T>
inline ThreadVariable<T>::ThreadVariable()
    : changedOutside(false)
{}

template <class T>
inline ThreadVariable<T>::ThreadVariable(const T& t)
    : data(t)
    , changedOutside(true)
{}

template <class T>
inline ThreadVariable<T>& ThreadVariable<T>::operator=(const T& t)
{
    {
        std::unique_lock<std::mutex> lock(mutex);
        data = t;
        changedOutside = true;
    }
    return *this;
}

template <class T>
inline bool ThreadVariable<T>::CheckChanged(T& newData)
{
    std::unique_lock<std::mutex> lock(mutex);
    newData = data;
    if(changedOutside)
    {
        changedOutside = false;
        return true;
    }
    return false;
}

template <class T>
inline T ThreadVariable<T>::Get() const
{
    std::unique_lock<std::mutex> lock(mutex);
    return data;
}