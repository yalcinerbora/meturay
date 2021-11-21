template<class T>
bool MPMCQueue<T>::Empty()
{
    return (dequeueLoc + 1) % data.size()  == enqueueLoc;
}

template<class T>
bool MPMCQueue<T>::Full()
{
    return enqueueLoc == dequeueLoc;
}

template<class T>
void MPMCQueue<T>::Increment(size_t& i)
{
    i += 1;
    i %= data.size();
}

template<class T>
MPMCQueue<T>::MPMCQueue(size_t bufferSize)
    : data(bufferSize)
    , enqueueLoc(1)
    , dequeueLoc(0)
    , terminate(false)
{
    assert(bufferSize > 1);
}

template<class T>
void MPMCQueue<T>::Dequeue(T& item)
{
    {
        std::unique_lock<std::mutex> lock(mutex);
        dequeueWake.wait(lock, [&]()
        {
            return !Empty();
        });

        Increment(dequeueLoc);
        item = std::move(data[dequeueLoc]);
    }
    enqueWake.notify_one();
}

template<class T>
bool MPMCQueue<T>::TryDequeue(T& item)
{
    {
        std::unique_lock<std::mutex> lock(mutex);
        if(Empty() || terminate) return false;

        Increment(dequeueLoc);
        item = std::move(data[dequeueLoc]);
    }
    enqueWake.notify_one();
    return true;
}

template<class T>
void MPMCQueue<T>::Enqueue(T&& item)
{
    {
        std::unique_lock<std::mutex> lock(mutex);
        enqueWake.wait(lock, [&]()
        {
            return !Full() || terminate;
        });

        data[enqueueLoc] = std::move(item);
        Increment(enqueueLoc);
    }
    dequeueWake.notify_one();
}

template<class T>
bool MPMCQueue<T>::TryEnqueue(T&& item)
{
    {
        std::unique_lock<std::mutex> lock(mutex);
        if(Full() || terminate) return false;

        data[enqueueLoc] = std::move(item);
        Increment(enqueueLoc);
    }
    dequeueWake.notify_one();
    return true;
}

template<class T>
void MPMCQueue<T>::Terminate()
{
    {
        std::unique_lock<std::mutex> lock(mutex);
        terminate = true;
    }
    dequeueWake.notify_all();
    enqueWake.notify_all();
}