#pragma once

/**

Fix Sized Multi-producer Multi-consumer ring buffer queue,
shitty implementation (single lock and cond var used)

TODO: improve, atomic etc...

*/

#include <vector>
#include <mutex>
#include <condition_variable>
#include <cassert>

template<class T>
class MPMCQueue
{
    private:
        std::vector<T>          data;

        size_t                  enqueueLoc;
        size_t                  dequeueLoc;

        std::condition_variable enqueWake;
        std::condition_variable dequeueWake;

        std::mutex              mutex;
        bool                    terminate;

        bool                    Empty();
        bool                    Full();
        void                    Increment(size_t&);

    protected:
    public:
        // Constructors & Destructor
                                MPMCQueue(size_t bufferSize);
                                MPMCQueue(const MPMCQueue&) = delete;
                                MPMCQueue(MPMCQueue&&) = delete;
        MPMCQueue&              operator=(const MPMCQueue&) = delete;
        MPMCQueue&              operator=(MPMCQueue&&) = delete;
                                ~MPMCQueue() = default;

        // Interface
        void                    Dequeue(T&);
        bool                    TryDequeue(T&);
        void                    Enqueue(T&&);
        bool                    TryEnqueue(T&&);

        // Awakes all threads and forces them to leave queue
        void                    Terminate();
};

#include "MPMCQueue.hpp"