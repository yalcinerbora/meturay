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

## License

This Project is licensed under the MIT license. Please check [LICENSE](LICENSE) for details.





