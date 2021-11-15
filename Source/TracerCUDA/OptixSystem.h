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
            using OptixDevice = std::pair<CudaGPU&, OptixDeviceContext>;

            struct OptixState
            {
                OptixDeviceContext  context  = nullptr;
                OptixModule         module   = nullptr;
                OptixPipeline       pipeline = nullptr;
            };

        private:
            const CudaSystem&                   cudaSystem;
            std::vector<OptixState>             deviceStates;

            std::vector<OptixDevice>            optixDevices;

            static void                         OptixLOG(unsigned int level,
                                                         const char* tag,
                                                         const char* message,
                                                         void*);

            static TracerError                  LoadPTXFile(std::string& ptxSource,
                                                            const CudaGPU&,
                                                            const std::string& baseName);

        protected:
        public:
            // Constructors & Destructor
                                            OptiXSystem(const CudaSystem&);
                                            OptiXSystem(const OptiXSystem&) = delete;
            OptiXSystem&                    operator=(const OptiXSystem&) = delete;
                                            ~OptiXSystem();

            const std::vector<OptixDevice>& OptixCapableDevices() const;

            TracerError                     OptixGenerateModules(const OptixModuleCompileOptions&,
                                                                 const OptixPipelineCompileOptions&,
                                                                 const std::string& baseFileName);
            TracerError                     OptixGeneratePipelines(const OptixPipelineCompileOptions& pOpts,
                                                                   const OptixPipelineLinkOptions& lOpts,
                                                                   const std::vector<OptixProgramGroup>& programs);

            //
            OptixDeviceContext              OptixContext(const CudaGPU&) const;
            OptixModule                     OptixModule(const CudaGPU&) const;


    };
#endif
#endif