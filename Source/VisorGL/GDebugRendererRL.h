#pragma once

#include "GDebugRendererI.h"

#include <nlohmann/json_fwd.hpp>
#include <GL/glew.h>

#include "RayLib/AABB.h"
#include "ShaderGL.h"
#include "TextureGL.h"

struct SurfaceLeaf
{
    Vector3f position;
    Vector3f normal;
    uint32_t leafId;
};

struct /*alignas(16)*/ SurfaceLBVHNode
{
    // Pointers
    union
    {
        // Non-leaf part
        struct
        {
            Vector3 aabbMin;
            uint32_t left;
            Vector3 aabbMax;
            uint32_t right;
        } body;
        // leaf part
        SurfaceLeaf leaf;
    };
    uint32_t    parent;
    bool        isLeaf;
};

struct SurfaceLBVH
{
    std::vector<SurfaceLBVHNode>    nodes;
    uint32_t                        nodeCount;
    uint32_t                        leafCount;
    uint32_t                        rootIndex;

    uint32_t FindNearestPoint(float& distance, const Vector3f& worldPoint) const;
    float    VoronoiCenterSize() const;
};

class GDebugRendererRL : public GDebugRendererI
{
    public:
        static constexpr const char* TypeName = "RL";

        // Shader Bind Points
        // SSBOs
        static constexpr GLuint     SSB_MAX_LUM = 0;
        // UBOs
        static constexpr GLuint     UB_MAX_LUM = 0;
        // Uniforms
        static constexpr GLuint     U_RES = 0;
        static constexpr GLuint     U_LOG_ON = 1;
        // Textures
        static constexpr GLuint     T_IN_LUM_TEX = 0;
        static constexpr GLuint     T_IN_GRAD_TEX = 1;
        // Images
        static constexpr GLuint     I_OUT_REF_IMAGE = 0;

    private:
        static constexpr const char* LBVH_NAME = "lbvh";
        static constexpr const char* QFUNC_NAME = "qFunctions";
        static constexpr const char* QSIZE_NAME = "qSize";

        const SamplerGL             linearSampler;
        const TextureGL&            gradientTexture;
        uint32_t                    curLocationIndex;
        // Name of the Guider (shown in GUI)
        std::string                 name;
        //
        TextureGL                   currentTexture;
        std::vector<float>          currentValues;
        float                       maxValueDisplay;
        // Shaders
        ShaderGL                    compReduction;
        ShaderGL                    compRefRender;

        // Spatial Data Structure
        SurfaceLBVH                 lbvh;
        // Directional Data Structures
        std::vector<std::string>    qFuncFileNames;
        Vector2ui                   qFuncSize;
        // Options
        bool                        renderPerimeter;

        static bool                 LoadLBVH(SurfaceLBVH&,
                                             const nlohmann::json& config,
                                             const std::string& configPath);

    protected:

    public:
        // Constructors & Destructor
                            GDebugRendererRL(const nlohmann::json& config,
                                             const TextureGL& gradientTexture,
                                             const std::string& configPath,
                                             uint32_t depthCount);
                            GDebugRendererRL(const GDebugRendererRL&) = delete;
        GDebugRendererRL&   operator=(const GDebugRendererRL&) = delete;
                            ~GDebugRendererRL();

        // Interface
        void                RenderSpatial(TextureGL&, uint32_t depth,
                                          const std::vector<Vector3f>& worldPositions) override;
        void                UpdateDirectional(const Vector3f& worldPos,
                                              bool doLogScale,
                                              uint32_t depth) override;

        bool                RenderGUI(bool& overlayCheckboxChanged,
                                      bool& overlayValue,
                                      const ImVec2& windowSize) override;
};