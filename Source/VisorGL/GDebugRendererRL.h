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
};

struct alignas(16) SurfaceLBVHNode
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
    std::vector<SurfaceLBVHNode> nodes;
    uint32_t nodeCount;
    uint32_t leafCount;
    uint32_t rootIndex;

    uint32_t FindNearestPoint(float& distance, const Vector3f& worldPoint) const;
    float    VoronoiCenterSize() const;
};

class GDebugRendererRL : public GDebugRendererI
{
    public:
        static constexpr const char* TypeName = "RL";

    private:
        static constexpr const char* LBVH_NAME = "lbvh";
        static constexpr const char* QFUNC_NAME = "qFunctions";

        const SamplerGL         linearSampler;
        const TextureGL&        gradientTexture;
        uint32_t                curLocationIndex;
        // Name of the Guider (shown in GUI)
        std::string             name;
        //
        TextureGL               currentTexture;
        std::vector<float>      currentValues;
        float                   maxValueDisplay;

        // Spatial Data Structure
        SurfaceLBVH             lbvh;
        // Directional Data Structures
        // ....
        // Options
        bool                    renderPerimeter;

        static bool             LoadLBVH(SurfaceLBVH&,
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