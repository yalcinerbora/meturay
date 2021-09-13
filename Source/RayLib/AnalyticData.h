#pragma once

struct AnalyticData
{
    // Performance
    double      raysPerSec;
    // Memory Related
    double      TotalGPUMemory; // MiB
    double      totalCPUMemoru; // MiB

    // Kernel Utilization
    double      rayPerKernelMaterial;
    double      rayPerKernelAccelerator;

    // Timings
    double      lastSceneLoadTime;      // secs
    double      lastSceneUpdateTime;    // secs
};
