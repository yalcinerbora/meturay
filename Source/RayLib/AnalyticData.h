#pragma once

struct SceneAnalyticData
{
    enum SceneGroupTypes
    {
        MATERIAL,
        PRIMITIVE,
        LIGHT,
        CAMERA,
        ACCELERATOR,
        TRANSFORM,
        MEDIUM,

        END
    };

    // Generic
    std::string                 sceneName;
    // Timings
    double                      sceneLoadTime;      // secs
    double                      sceneUpdateTime;    // secs
    // Group Counts
    std::array<uint32_t, END>   groupCounts;
};

struct AnalyticData
{
    // Performance
    double          throughput;
    std::string     throughputName;
    double          time;
    // Timings


    // GPU Related
    double      workGroupCount;
    // Memory Related
    double      TotalGPUMemoryMiB; // MiB
    double      totalCPUMemoruMiB; // MiB

};
