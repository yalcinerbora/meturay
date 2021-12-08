#pragma once
/**

Worker Implementation

Basic worker implementation. Single thread and
a job queue that assigns any function to the worker.

*/
#include <queue>
#include <mutex>
#include <thread>
#include <future>
#include <functional>
#include <condition_variable>

// TODO: there is no limit on the queue
// it may be an issue..
class WorkerThread
{
    private:
        std::thread                         workerThread;
        bool                                stopSignal;

        // Queue and Associated Helpers
        std::queue<std::function<void()>>   assignedJobs;
        mutable std::mutex                  mutex;
        mutable std::condition_variable     conditionVar;

        // Entry Point of the thread
        void                                THRDEntryPoint();
        bool                                ProcessJob();

    protected:

    public:
        // Constructor & Destructor
                                            WorkerThread();
                                            WorkerThread(const WorkerThread&) = delete;
        WorkerThread&                       operator=(const WorkerThread&) = delete;
                                            ~WorkerThread() = default;

        // ThreadLifetime Worker
        void                                Start();
        void                                Stop();

        // Return the Amount of Queued Jobs
        int                                 QueuedJobs() const;

        // Function Def Copied From std::async
        template <class Function, class... Args>
        std::future<typename std::invoke_result<Function(Args...)>::type>
                                            AddWork(Function&&, Args&&...);
};

// Template Functions
template <class Function, class... Args>
std::future<typename std::invoke_result<Function(Args...)>::type>
WorkerThread::AddWork(Function&& f, Args&&... args)
{
    typedef typename std::invoke_result<Function(Args...)>::type returnType;

    // I had to make this by using make_shared
    // I tried to make this without make_shared since it is extra operation but w/e
    auto job = std::make_shared<std::packaged_task<returnType()>>(std::bind(f, args...));
    auto future = job->get_future();
    auto jobWrap = [job]()
    {
        (*job)();
    };

    mutex.lock();
    assignedJobs.push(jobWrap);
    mutex.unlock();
    conditionVar.notify_all();
    return future;
}