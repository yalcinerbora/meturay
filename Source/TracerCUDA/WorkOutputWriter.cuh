#pragma once

template<class AuxType>
class OutputWriter
{
    public:
        static constexpr uint32_t   SupportedMaxOut = 8;

    private:
        bool                    isWritten[SupportedMaxOut];
        RayGMem*                gOutRays;
        AuxType*                gOutAux;
        HitKey*                 gOutBoundKeys;
        uint32_t                maxOut;

    public:
        // Constructors & Destructor
        __device__          OutputWriter(HitKey* gOutBoundKeys,
                                         RayGMem* gOutRays,
                                         AuxType* gOutAux,
                                         uint32_t maxOut);
                            OutputWriter(const OutputWriter&) = delete;
        OutputWriter&       operator=(const OutputWriter&) = delete;
        __device__          ~OutputWriter();

        // Methods
        __device__ void     Write(uint32_t index,
                                  const RayReg&,
                                  const AuxType&);
        __device__ void     Write(uint32_t index,
                                  const RayReg&,
                                  const AuxType&,
                                  const HitKey&);
};

template<class AuxType>
__device__ __forceinline__
OutputWriter<AuxType>::OutputWriter(HitKey* gOutBoundKeys,
                                    RayGMem* gOutRays,
                                    AuxType* gOutAux,
                                    uint32_t maxOut)
    : isWritten{false, false, false, false,
                false, false, false, false}
    , gOutRays(gOutRays)
    , gOutAux(gOutAux)
    , gOutBoundKeys(gOutBoundKeys)
    , maxOut(maxOut)
{}

template<class AuxType>
__device__ __forceinline__
OutputWriter<AuxType>::~OutputWriter()
{
    static constexpr RayReg INV_RAY = EMPTY_RAY_REGISTER;
    for(uint32_t i = 0; i < maxOut; i++)
    {
        if(isWritten[i]) continue;
        // Generate Dummy Ray and Terminate
        INV_RAY.Update(gOutRays, i);
        gOutBoundKeys[i] = HitKey::InvalidKey;
    }
}

template<class AuxType>
__device__ __forceinline__
void OutputWriter<AuxType>::Write(uint32_t index,
                                  const RayReg& r,
                                  const AuxType& aux)
{
    assert(index < maxOut);
    isWritten[index] = true;

    r.Update(gOutRays, index);
    gOutAux[index] = aux;
}

template<class AuxType>
__device__ __forceinline__
void OutputWriter<AuxType>::Write(uint32_t index,
                                  const RayReg& r,
                                  const AuxType& aux,
                                  const HitKey& hk)
{
    assert(index < maxOut);
    isWritten[index] = true;

    r.Update(gOutRays, index);
    gOutAux[index] = aux;
    gOutBoundKeys[index] = hk;
}