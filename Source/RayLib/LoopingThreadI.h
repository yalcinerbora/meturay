#pragma once
/**

Looping Thread Partial Interface

used by threads that do same work over and over again
like tracer thread and visor thread.

Visor thread continuously renders stuff until terminated
Tracer(s) continuously render stuff until terminated

User can define internal terminate condition where thread automatically ends

*/

#include <thread>
#include <mutex>
#include <condition_variable>

class LoopingThreadI
{
        private:
        std::thread                 thread;
        std::mutex                  mutex;
        std::condition_variable     conditionVar;
        bool                        stopSignal;
        bool                        pauseSignal;

        void                        THRDEntry();

    protected:
        virtual bool                InternallyTerminated() const = 0;
        virtual void                InitialWork() = 0;
        virtual void                LoopWork() = 0;
        virtual void                FinalWork() = 0;

    public:
        // Constructors & Destructor
                                    LoopingThreadI();
        virtual                     ~LoopingThreadI();

        void                        Start();
        void                        Stop();
        void                        Pause(bool pause);

        bool                        IsTerminated();
};

inline void LoopingThreadI::THRDEntry()
{
    InitialWork();

    while(!InternallyTerminated() || stopSignal)
    {
        LoopWork();

        // Condition
        {
            std::unique_lock<std::mutex> lock(mutex);
            // Wait if paused (or stop signal is set)
            conditionVar.wait(lock, [&]
            {
                return stopSignal || !pauseSignal;
            });
        }

        // Break the thread loop and terminate
        if(stopSignal) break;
    }
    FinalWork();
}

inline LoopingThreadI::LoopingThreadI()
    : stopSignal(false)
    , pauseSignal(false)
{}

inline LoopingThreadI::~LoopingThreadI()
{
    Stop();
}

inline void LoopingThreadI::Start()
{
    // If the system was previously paused
    // ignore that
    pauseSignal = false;

    // Launch a new thread
    thread = std::thread(&LoopingThreadI::THRDEntry, this);
}

inline void LoopingThreadI::Stop()
{
    mutex.lock();
    stopSignal = true;
    mutex.unlock();
    conditionVar.notify_one();
    if(thread.joinable()) thread.join();
    stopSignal = false;
}

inline void LoopingThreadI::Pause(bool pause)
{
    mutex.lock();
    pauseSignal = pause;
    mutex.unlock();
    conditionVar.notify_one();
}

inline bool LoopingThreadI::IsTerminated()
{
    return /*!thread.joinable() ||*/ InternallyTerminated();
}