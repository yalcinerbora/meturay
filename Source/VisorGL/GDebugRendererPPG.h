#pragma once


#include "GDebugRendererI.h"

#include <nlohmann/json_fwd.hpp>


struct DTree
{

};

class GDebugRendererPPG : public GDebugRendererI
{
    public:
        static constexpr const char* TypeName = " PPG";

    private:
        const TextureGL&    gradientTexture;

    protected:
   
    public:
        // Constructors & Destructor
                            GDebugRendererPPG(const nlohmann::json& config,
                                              const TextureGL& gradientTexture);
                            GDebugRendererPPG(const GDebugRendererPPG&) = delete;
        GDebugRendererPPG&  operator=(const GDebugRendererPPG&) = delete;
                            ~GDebugRendererPPG() = default;

        // Interface
        void                RenderSpatial(TextureGL&, uint32_t depth) override;
        void                RenderDirectional(TextureGL&, const Vector3f& worldPos, 
                                              uint32_t depth) override;

};
