#include "GDebugRendererPPG.h"


GDebugRendererPPG::GDebugRendererPPG(const nlohmann::json& config,
                                     const TextureGL& gradientTexture)
    : gradientTexture(gradientTexture)
{

}

void GDebugRendererPPG::RenderSpatial(TextureGL&, uint32_t depth)
{

}

void GDebugRendererPPG::RenderDirectional(TextureGL&, const Vector3f& worldPos, uint32_t depth)
{

}