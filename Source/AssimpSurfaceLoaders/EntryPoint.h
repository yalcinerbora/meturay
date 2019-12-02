#pragma once

class SurfaceLoaderPoolI;

extern "C" _declspec(dllexport) SurfaceLoaderPoolI * __stdcall GenerateAssimpPool();

extern "C" _declspec(dllexport) void __stdcall DeleteAssimpPool(SurfaceLoaderPoolI * tGen);