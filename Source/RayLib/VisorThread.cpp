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
    // Set Rendering context for this thread
    visor.SetRenderingContextCurrent();
}

void VisorThread::LoopWork()
{
    visor.Render();
}

void VisorThread::FinalWork()
{
    // No final work for Visor
    visor.ReleaseRenderingContext();
}

void VisorThread::AccumulateImagePortion(const std::vector<Byte> data,
                                         PixelFormat f, size_t offset,
                                         Vector2i start, Vector2i end)
{
    // Visor itself has thread safe queue for these operations
    // Directly delegate
    visor.AccumulatePortion(data, f, offset, start, end);
}

void VisorThread::ProcessInputs()
{
    visor.ProcessInputs();
}