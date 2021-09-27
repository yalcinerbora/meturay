#pragma once

#include <string>
#include <vector>
#include "RayLib/Vector.h"
#include <nlohmann/json_fwd.hpp>

#include "TextureGL.h"

class GDebugRendererRef
{
    public:
        using PathList = std::vector<std::string>;

        static constexpr const char*  RESOLUTION_NAME = "resolution";
        static constexpr const char* IMAGES_NAME = "images";

    private:
        Vector2i            resolution;
        PathList            referencePaths;

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
                                                  const Vector2i& pixel,
                                                  const Vector2i& resolution) const;

};