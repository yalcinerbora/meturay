template<class T>
bool MPMCQueue<T>::IsEmpty()
{
    return ((dequeueLoc + 1) % data.size())  == enqueueLoc;
}

template<class T>
bool MPMCQueue<T>::IsFull()
{
    return enqueueLoc == dequeueLoc;
}

template<class T>
void MPMCQueue<T>::Increment(size_t& i)
{
    i = (i + 1) % data.size();
}

template<class T>
MPMCQueue<T>::MPMCQueue(size_t bufferSize)
    : data(bufferSize + 1)
    , enqueueLoc(1)
    , dequeueLoc(0)
    , terminate(false)
{}

template<class T>
void MPMCQueue<T>::Dequeue(T& item)
{
    if (terminate) return;
    {
        std::unique_lock<std::mutex> lock(mutex);
        dequeueWake.wait(lock, [&]()
        {
            return (!IsEmpty() || terminate);
        });
        if (terminate) return;

        Increment(dequeueLoc);
        item = std::move(data[dequeueLoc]);
    }
    enqueWake.notify_one();
}

template<class T>
bool MPMCQueue<T>::TryDequeue(T& item)
{
    if (terminate) return false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        if(IsEmpty() || terminate) return false;

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
            return (!IsFull() || terminate);
        });
        if (terminate) return;

        data[enqueueLoc] = std::move(item);
        Increment(enqueueLoc);
    }
    dequeueWake.notify_one();
}

template<class T>
bool MPMCQueue<T>::TryEnqueue(T&& item)
{
    if (terminate) return;
    {
        std::unique_lock<std::mutex> lock(mutex);
        if(IsFull() || terminate) return false;

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