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

    // Texture Cache related
    double      texCacheHitRateGPU;     // Percentage of Hits on GPU Memory
    double      texCacheHitRateCPU;     // Percentage of Hits on CPU Memory
    double      texCacheHitRateHDD;     // Percentage of Hits on Cold Memory

    // Timings
    double      lastSceneLoadTime;      // secs
    double      lastSceneUpdateTime;    // secs
};
