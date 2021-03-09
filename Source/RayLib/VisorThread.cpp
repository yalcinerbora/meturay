#include "VisorThread.h"
#include "VisorI.h"

#include "Log.h"

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

void VisorThread::AccumulateImagePortion(const std::vector<Byte> data,
                                         PixelFormat f, size_t offset,
                                         Vector2i start, Vector2i end)
{
    // Visor itself has thread safe queue for these operations
    // Directly delegate
    visor.AccumulatePortion(std::move(data), f, offset, start, end);
}

void VisorThread::ProcessInputs()
{
    visor.ProcessInputs();
}