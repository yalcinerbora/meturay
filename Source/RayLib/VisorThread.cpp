#include "VisorThread.h"
#include "VisorI.h"

VisorThread::VisorThread(VisorI& v)
    : visor(v)
{}

bool VisorThread::InternallyTerminated() const
{
    return !visor.IsOpen();
}

void VisorThread::InitialWork()
{
    // No initial work for Visor
}

void VisorThread::LoopWork()
{
    visor.Render();
}

void VisorThread::FinalWork() 
{
    // No final work for Visor
}
