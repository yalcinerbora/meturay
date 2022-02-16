#pragma once


#include "GDebugRendererI.h"

#include <nlohmann/json_fwd.hpp>
#include <GL/glew.h>

#include "RayLib/AABB.h"
#include "ShaderGL.h"
#include "TextureGL.h"

struct DTreeNodeCPU
{
    uint32_t    parentIndex;
    Vector4ui   childIndices;
    Vector4f    irradianceEstimates;
    Vector4ui   sampleCounts;

    bool IsRoot() const
    {
        return parentIndex == UINT32_MAX;
    }

    bool IsLeaf(uint8_t childId) const
    {
        return childIndices[childId] == UINT32_MAX;
    }
};

struct STreeNode
{
    enum class AxisType : int8_t
    {
        X = 0,
        Y = 1,
        Z = 2,

        END
    };

    AxisType    splitAxis; // In which dimension this node is split
    bool        isLeaf;    // Determines which data the index is holding

    // It is either DTree index or next child index
    // Children are always grouped (children + 1 is the other child)
    uint32_t    index;

    bool DetermineChild(const Vector3f& normalizedCoords) const
    {
        // Binary tree is always mid split so check half
        return normalizedCoords[static_cast<int>(splitAxis)] >= 0.5f;
    }

    Vector3f NormalizeCoordsForChild(bool leftRight, const Vector3f& parentNormalizedCoords) const
    {
        Vector3f result = parentNormalizedCoords;
        int axis = static_cast<int>(splitAxis);
        if(leftRight) result[axis] -= 0.5f;
        result[axis] *= 2.0f;
        return result;
    }
};

struct SDTree
{
    AABB3f                                  extents;
    std::vector<STreeNode>                  sTreeNodes;
    std::vector<std::vector<DTreeNodeCPU>>  dTreeNodes;
    std::vector<std::pair<uint32_t, float>> dTrees;

    uint32_t FindDTree(const Vector3f& worldPos) const
    {
        uint32_t dTreeIndex = UINT32_MAX;
        if(sTreeNodes.size() == 0) return dTreeIndex;

        // Convert to Normalized Tree Space
        Vector3f normalizedCoords = worldPos - extents.Min();
        normalizedCoords /= (extents.Max() - extents.Min());

        const STreeNode* node = sTreeNodes.data();
        while(true)
        {
            if(node->isLeaf)
            {
                dTreeIndex = node->index;
                break;
            }
            else
            {
                bool leftRight = node->DetermineChild(normalizedCoords);
                normalizedCoords = node->NormalizeCoordsForChild(leftRight, normalizedCoords);
                // Traverse...
                node = sTreeNodes.data() + node->index + ((leftRight) ? 0 : 1);
            }
        }
        return dTreeIndex;
    }
};

class GDebugRendererPPG : public GDebugRendererI
{
    public:
        static constexpr const char* TypeName = "PPG";

        // Shader Binding Locations
        // Vertex In (Per Vertex)
        static constexpr GLenum     IN_POS = 0;
        // Vertex In  (Per Instance)
        static constexpr GLenum     IN_OFFSET = 1;
        static constexpr GLenum     IN_DEPTH = 2;
        static constexpr GLenum     IN_RADIANCE = 3;
        // Uniforms
        static constexpr GLenum     U_MAX_RADIANCE = 0;
        static constexpr GLenum     U_PERIMIETER_ON = 1;
        static constexpr GLenum     U_PERIMIETER_COLOR = 2;
        static constexpr GLenum     U_MAX_DEPTH = 3;
        static constexpr GLenum     U_LOG_ON = 4;

        static constexpr GLenum     U_RES = 0;
        static constexpr GLenum     U_AABB_MIN = 1;
        static constexpr GLenum     U_AABB_MAX = 2;
        static constexpr GLenum     U_NODE_COUNT = 3;
        // Shader Storage Buffers
        static constexpr GLenum     SSB_STREE = 0;
        static constexpr GLenum     SSB_LEAF_COL = 1;
        static constexpr GLenum     SSB_WORLD_POS = 2;
        // Images
        static constexpr GLenum     I_OUT_IMAGE = 0;
        // Textures
        static constexpr GLenum     T_IN_GRADIENT = 0;
        // FBO Outputs
        static constexpr GLenum     OUT_COLOR = 0;
        static constexpr GLenum     OUT_VALUE = 1;

    private:
        static constexpr const char* SD_TREE_NAME = "sdTrees";

        const SamplerGL         linearSampler;
        const TextureGL&        gradientTexture;
        uint32_t                curDTreeIndex;
        // All SD Trees that are loaded
        std::vector<SDTree>     sdTrees;
        // Color of the perimeter (In order to visualize D-Trees Properly
        Vector3f                perimeterColor;
        // Name of the Guider (shown in GUI)
        std::string             name;
        //
        TextureGL               currentTexture;
        std::vector<float>      currentValues;
        float                   maxValueDisplay;
        // Options
        bool                    renderPerimeter;
        bool                    renderSamples;

        // OGL Related
        // FBO (Since we use raster pipeline to render)
        // VAO etc..
        GLuint                  fbo;
        GLuint                  vao;
        GLuint                  indexBuffer;
        GLuint                  vPosBuffer;
        GLuint                  treeBuffer;
        size_t                  treeBufferSize;

        ShaderGL                vertDTreeRender;
        ShaderGL                fragDTreeRender;
        ShaderGL                compSTreeRender;

        static bool             LoadSDTree(SDTree&,
                                           const nlohmann::json& config,
                                           const std::string& configPath,
                                           uint32_t depth);

    protected:

    public:
        // Constructors & Destructor
                            GDebugRendererPPG(const nlohmann::json& config,
                                              const TextureGL& gradientTexture,
                                              const std::string& configPath,
                                              uint32_t depthCount);
                            GDebugRendererPPG(const GDebugRendererPPG&) = delete;
        GDebugRendererPPG&  operator=(const GDebugRendererPPG&) = delete;
                            ~GDebugRendererPPG();

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
