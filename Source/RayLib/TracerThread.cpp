#include "TracerThread.h"
#include "GPUTracerI.h"

TracerThread::TracerThread(GPUTracerI& t)
    : tracer(t)
{}

bool TracerThread::InternallyTerminated() const
{

}

void TracerThread::InitialWork()
{    
    // No initial work for tracer
}

void TracerThread::LoopWork()
{
    // Generate Initial Work
    // Select a camera
    // TODO: ..................
    tracer.GenerateWork(0);

    // Render
    while(tracer.Render());
    // Finalize (send image to visor)
    tracer.Finalize();
}

void TracerThread::FinalWork()
{
    // No final work for tracer
    // Eveything should destroy gracefully
}