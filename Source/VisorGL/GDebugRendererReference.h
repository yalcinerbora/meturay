#pragma once

#include <string>
#include <vector>
#include <nlohmann/json_fwd.hpp>

#include "RayLib/Vector.h"
#include "TextureGL.h"
#include "ShaderGL.h"

class GDebugRendererRef
{
    public:
        using PathList = std::vector<std::string>;

        static constexpr const char*  RESOLUTION_NAME = "resolution";
        static constexpr const char* IMAGES_NAME = "images";

        // Shader Bind Points
        // SSBOs
        static constexpr GLuint     SSB_MAX_LUM = 0;
        // UBOs
        static constexpr GLuint     UB_MAX_LUM = 0;
        // Uniforms
        static constexpr GLuint     U_RES = 0;
        // Textures
        static constexpr GLuint     T_IN_LUM_TEX = 0;
        static constexpr GLuint     T_IN_GRAD_TEX = 1;
        // Images
        static constexpr GLuint     I_OUT_REF_IMAGE = 0;

    private:
        Vector2i            resolution;
        PathList            referencePaths;

        ShaderGL            compReduction;
        ShaderGL            compRefRender;

        SamplerGL           linearSampler;

        const TextureGL&    gradientTex;

        void                LoadPaths(const Vector2i& resolution,
                                      const std::string& pathRegex);

    protected:
    public:
        // Constructors & Destructor
                                GDebugRendererRef() = default;
                                GDebugRendererRef(const nlohmann::json& config,
                                                  const TextureGL& gradientTex);
                                GDebugRendererRef(const GDebugRendererRef&) = delete;
                                GDebugRendererRef(GDebugRendererRef&&) = default;
        GDebugRendererRef&      operator=(const GDebugRendererRef&) = delete;
        GDebugRendererRef&      operator=(GDebugRendererRef&&) = default;
                                ~GDebugRendererRef() = default;

        //
        void                    RenderSpatial(TextureGL&) const;
        void                    RenderDirectional(TextureGL&,
                                                  std::vector<float>& values,
                                                  const Vector2i& worldPixel,
                                                  const Vector2i& worldResolution) const;
};