#include "WorkerThread.h"

WorkerThread::WorkerThread()
    : stopSignal(false)
{}

void WorkerThread::THRDEntryPoint()
{
    // Thread Entry Point
    while(ProcessJob());
}

bool WorkerThread::ProcessJob()
{
    std::function<void()> func;

    // Handling Job Queue pop
    {
        std::unique_lock<std::mutex> lock(mutex);

        // Wait if queue is empty
        conditionVar.wait(lock, [&]
        {
            return stopSignal || !assignedJobs.empty();
        });

        // Exit if Stop is Signaled and There is no other
        if(stopSignal && assignedJobs.empty())
            return false;

        func = assignedJobs.front();
        assignedJobs.pop();
    }

    // Actual function call (out of critical section)
    func();
    return true;
}

void WorkerThread::Start()
{
    workerThread = std::thread(&WorkerThread::THRDEntryPoint, this);
}

void WorkerThread::Stop()
{
    mutex.lock();
    stopSignal = true;
    mutex.unlock();
    conditionVar.notify_one();
    workerThread.join();
    stopSignal = false;
}

int WorkerThread::QueuedJobs() const
{
    std::unique_lock<std::mutex> lock(mutex);
    return static_cast<int>(assignedJobs.size());
}