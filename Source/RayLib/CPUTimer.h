#pragma once
/**

std::chrono wrapper for high performance time querying
in a single thread.

*/

#include <chrono>

// JESUS THAT NAMESPACES+TEMPLATES
using CPUClock = std::chrono::high_resolution_clock;
using CPUTimePoint = std::chrono::time_point<CPUClock>;
using CPUDuration = CPUTimePoint::duration;

using CPUTimeNanos = std::nano;
using CPUTimeMicros = std::micro;
using CPUTimeMillis = std::milli;
using CPUTimeSeconds = std::ratio<1>;
using CPUTimeMins = std::ratio<60>;

namespace Utility
{
class CPUTimer
{
    private:
        CPUDuration     elapsed;
        CPUTimePoint    start;

    protected:
    public:
        // Constructors & Destructor
                        CPUTimer() = default;
                        ~CPUTimer() = default;

        // Utility
        void            Start();
        void			Stop();
        void			Lap();

        template <class Time>
        double          Elapsed();
};

inline void CPUTimer::Start()
{
    start = CPUClock::now();
}

inline void CPUTimer::Stop()
{
    CPUTimePoint end = CPUClock::now();
    elapsed = end - start;
}

inline void CPUTimer::Lap()
{
    CPUTimePoint end = CPUClock::now();
    elapsed = end - start;
    start = end;
}

template <class Time>
double CPUTimer::Elapsed()
{
    return std::chrono::duration<double, Time>(elapsed).count();
}
}