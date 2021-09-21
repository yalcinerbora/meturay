#pragma once


#include "GDebugRendererI.h"

#include <nlohmann/json_fwd.hpp>
#include <GL/glew.h>

#include "RayLib/AABB.h"
#include "ShaderGL.h"
#include "TextureGL.h"

struct DTreeNode
{
    uint16_t    parentIndex;
    Vector4ui   childIndices;
    Vector4f    irradianceEstimates;
    
    bool IsRoot() const
    {
        return parentIndex == UINT16_MAX;
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
    // Childs are always grouped (childs + 1 is the other child)    
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
        if(leftRight) result[axis] -= 0.5;
        result[axis] *= 2.0f;
        return result;
    }
};

struct SDTree
{
    AABB3f                                  extents;
    std::vector<STreeNode>                  sTreeNodes;
    std::vector<std::vector<DTreeNode>>     dTreeNodes;
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
        
        // Textures
        static constexpr GLenum     T_IN_GRADIENT = 0;        
        // FBO Outputs
        static constexpr GLenum     OUT_COLOR = 0;
        static constexpr GLenum     OUT_VALUE = 1;

    private:
        static constexpr const char* SD_TREE_NAME = "SDTrees";

        const SamplerGL         linearSampler;
        const TextureGL&        gradientTexture;
        const std::string&      configPath;        
        uint32_t                depthCount;
        // All SD Trees that are loaded
        std::vector<SDTree>     sdTrees;
        // Color of the perimeter (In order to visualize D-Trees Properly
        Vector3f                perimeterColor;
        // Name of the Guider (shown in GUI
        std::string             name;
        // 
        std::vector<float>      irradianceValues;

        // OGL Related 
        // FBO (Since we use raster pipeline to render)
        // VAO etc..
        GLuint                  fbo;
        GLuint                  vao;
        GLuint                  vPos;
        GLuint                  indexBuffer;
        GLuint                  vPosBuffer;
        GLuint                  treeBuffer;
        size_t                  treeBufferSize;

        ShaderGL                vertDTreeRender;
        ShaderGL                fragDTreeRender;

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
        void                RenderSpatial(TextureGL&, uint32_t depth) override;
        void                RenderDirectional(TextureGL&,
                                              std::vector<float>& values,
                                              const Vector3f& worldPos, 
                                              uint32_t depth) override;

        const std::string&  Name() const override;
};