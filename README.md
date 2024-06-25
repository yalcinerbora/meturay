# METU Ray

METU Ray (MRay) - GPU-Based Renderer/Renderer Framework.

MRay is a ray tracing based renderer framework for researching computer graphics techniques. It is designed to be extensible for comparing different rendering methods, material (BxDF) sampling and ray accelerators. Although after hardware accelerated ray-tracing, the accelerator portion probably is not necessary.

## Features

- It is completely GPU-based for rapid rendering technique development

## Limitations

- Implemented using CUDA; thus, only CUDA-capable GPUs are supported. This means NVIDIA graphics cards only.
- Developed over Windows, it should work via MSVC. Time to time, the codebase compiled via the Linux compilers (clang, gcc) it do compile and run but it is not thoroughly tested.

## Building

- In its current form, most of the dependencies are statically compiled. This include:
    - assimp
    - FreeImage
    - glew
    - glfw
    - googletest
    - OpenEXR
    - fmtlib
    - spdlog
    - dearimgui

- This means only Visual Studio 2022 (up to latest supported version of MSVC) is supported. For Linux, the pre-compiled libraries should link.

- Other dependencies include:
    - CUDA: v11.X
    - NVIDIA Optix: v7.X

- These compilers/libraries should be installed by the user.

## If you are coming to this repo due to GRSI badge or from Paper[^1]

Please follow the instructions on [TO_RUN_FIG11](TO_RUN_FIG11). Powershell script "run_fig11.ps1" should generate the Figure 11 results.
Unfortunately, the codebase is not documented yet. For implementation details, you can check the "TracerCUDA/WFPGTracer*" source files.

[^1]:Bora Yalçıner, Ahmet Oğuz Akyüz; Path guiding for wavefront path tracing: A memory efficient approach for GPU path tracers; Computers & Graphics,
Volume 121, 2024, 103945, ISSN 0097-8493.

## License

This Project is licensed under the MIT license. Please check [LICENSE](LICENSE) for details.





