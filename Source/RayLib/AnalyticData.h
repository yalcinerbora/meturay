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
    // Key Maxiums
    Vector2i                    accKeyMax;
    Vector2i                    workKeyMax;
};

struct AnalyticData
{
    // Performance
    double          throughput;
    std::string     throughputSuffix;
    //
    double          workPerPixel;
    std::string     workPerPixelSuffix;
    // Timings
    float           iterationTime;      // msecs

    // GPU Related
    double          workGroupCount;
    // Memory Related
    double          totalGPUMemoryMiB; // MiB
    double          totalCPUMemoruMiB; // MiB

};