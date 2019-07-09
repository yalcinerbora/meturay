#pragma once


class GPUSceneI
{
    public:
    // Constructors & Destructor
    GPUScene(const std::string&,
             ScenePartitionerI& partitioner,
             TracerLogicGeneratorI&);
    GPUScene(const GPUScene&) = delete;
    GPUScene(GPUScene&&) noexcept;
    GPUScene& operator=(const GPUScene&) = delete;
    ~GPUScene();

    // Members
    size_t                      UsedGPUMemory();
    size_t                      UsedCPUMemory();
    //
    SceneError                  LoadScene(double);
    SceneError                  ChangeTime(double);
    //
    Vector2i                    MaxMatIds();
    Vector2i                    MaxAccelIds();
    HitKey                      BaseBoundaryMaterial();
    // Access GPU
    const LightStruct* LightsGPU() const;
    const TransformStruct* TransformsGPU() const;
    // Access CPU
    const CameraPerspective* CamerasCPU() const;
}