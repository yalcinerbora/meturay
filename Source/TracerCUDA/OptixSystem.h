#pragma once


#ifdef METU_CUDA
#ifdef MRAY_OPTIX
    #include <optix_host.h>
    #include <optix_stubs.h>
    #include <vector>
    #include "RayLib/TracerError.h"

    #include "DeviceMemory.h"

    static CUdeviceptr AsOptixPtr(DeviceMemory& mem)
    {
        return reinterpret_cast<CUdeviceptr>(static_cast<Byte*>(mem));
    }

    template<class T>
    static CUdeviceptr AsOptixPtr(T* ptr)
    {
        return reinterpret_cast<CUdeviceptr>(ptr);
    }

    class CudaSystem;
    class CudaGPU;

    class OptiXSystem
    {
        public:
            using OptixDevice = std::pair<const CudaGPU&, OptixDeviceContext>;

        private:
            const CudaSystem&                   cudaSystem;
            std::vector<OptixDevice>            optixDevices;

            static void                         OptixLOG(unsigned int level,
                                                         const char* tag,
                                                         const char* message,
                                                         void*);
        protected:
        public:
            // Constructors & Destructor
                                            OptiXSystem(const CudaSystem&);
                                            OptiXSystem(const OptiXSystem&) = delete;
            OptiXSystem&                    operator=(const OptiXSystem&) = delete;
                                            ~OptiXSystem();

            const std::vector<OptixDevice>& OptixCapableDevices() const;

            static TracerError              LoadPTXFile(std::string& ptxSource,
                                                        const CudaGPU&,
                                                        const std::string& baseName);

    };
#endif
#endif