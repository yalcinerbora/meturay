#pragma once

#include "GDebugRendererI.h"

#include <nlohmann/json_fwd.hpp>
#include <GL/glew.h>

#include "RayLib/AABB.h"
#include "RayLib/Ray.h"
#include "ShaderGL.h"
#include "TextureGL.h"

using IntegerNameList = std::vector<std::pair<uint32_t, std::string>>;
using Vec2uiNameList = std::vector<std::pair<Vector2ui, std::string>>;

// TODO: Make this "DRY"
struct SVOctree
{
    static constexpr uint32_t NORMAL_X_BIT_COUNT        = 9;
    static constexpr uint32_t NORMAL_Y_BIT_COUNT        = 9;
    static constexpr uint32_t NORMAL_LENGTH_BIT_COUNT   = 7;
    static constexpr uint32_t SPECULAR_BIT_COUNT        = 6;
    static constexpr uint32_t NORMAL_SIGN_BIT_COUNT     = 1;

    static_assert((NORMAL_X_BIT_COUNT +
                   NORMAL_Y_BIT_COUNT +
                   NORMAL_LENGTH_BIT_COUNT +
                   SPECULAR_BIT_COUNT +
                   NORMAL_SIGN_BIT_COUNT) == sizeof(uint32_t) * BYTE_BITS);

    static constexpr uint32_t NORMAL_X_BIT_MASK         = (1 << NORMAL_X_BIT_COUNT) - 1;
    static constexpr uint32_t NORMAL_Y_BIT_MASK         = (1 << NORMAL_Y_BIT_COUNT) - 1;
    static constexpr uint32_t NORMAL_LENGTH_BIT_MASK    = (1 << NORMAL_LENGTH_BIT_COUNT) - 1;
    static constexpr uint32_t SPECULAR_BIT_MASK         = (1 << SPECULAR_BIT_COUNT) - 1;
    static constexpr uint32_t NORMAL_SIGN_BIT_MASK      = (1 << NORMAL_SIGN_BIT_COUNT) - 1;

    static constexpr float UNORM_SPEC_FACTOR            = 1.0f / static_cast<float>(SPECULAR_BIT_COUNT);
    static constexpr float UNORM_LENGTH_FACTOR          = 1.0f / static_cast<float>(NORMAL_LENGTH_BIT_MASK);
    static constexpr float UNORM_NORM_X_FACTOR          = 1.0f / static_cast<float>(NORMAL_X_BIT_MASK);
    static constexpr float UNORM_NORM_Y_FACTOR          = 1.0f / static_cast<float>(NORMAL_Y_BIT_MASK);

    static constexpr uint32_t NORMAL_X_OFFSET           = 0;
    static constexpr uint32_t NORMAL_Y_OFFSET           = NORMAL_X_OFFSET + NORMAL_X_BIT_COUNT;
    static constexpr uint32_t NORMAL_LENGTH_OFFSET      = NORMAL_Y_OFFSET + NORMAL_Y_BIT_COUNT;
    static constexpr uint32_t SPECULAR_OFFSET           = NORMAL_LENGTH_OFFSET + NORMAL_LENGTH_BIT_COUNT;
    static constexpr uint32_t NORMAL_SIGN_BIT_OFFSET    = SPECULAR_OFFSET + SPECULAR_BIT_COUNT;

    static constexpr uint16_t LAST_BIT_UINT16 = (sizeof(uint16_t) * BYTE_BITS - 1);

    static constexpr uint64_t IS_LEAF_BIT_COUNT     = 1;
    static constexpr uint64_t CHILD_MASK_BIT_COUNT  = 8;
    static constexpr uint64_t PARENT_BIT_COUNT      = 28;
    static constexpr uint64_t CHILD_BIT_COUNT       = 27;

    static constexpr uint64_t CHILD_OFFSET          = 0;
    static constexpr uint64_t PARENT_OFFSET         = CHILD_OFFSET + CHILD_BIT_COUNT;
    static constexpr uint64_t CHILD_MASK_OFFSET     = PARENT_OFFSET + PARENT_BIT_COUNT;
    static constexpr uint64_t IS_LEAF_OFFSET        = CHILD_MASK_OFFSET + CHILD_MASK_BIT_COUNT;

    static constexpr uint64_t IS_LEAF_BIT_MASK      = (1 << IS_LEAF_BIT_COUNT) - 1;
    static constexpr uint64_t PARENT_BIT_MASK       = (1 << PARENT_BIT_COUNT) - 1;
    static constexpr uint64_t CHILD_BIT_MASK        = (1 << CHILD_BIT_COUNT) - 1;
    static constexpr uint64_t CHILD_MASK_BIT_MASK   = (1 << CHILD_MASK_BIT_COUNT) - 1;

    static constexpr uint32_t INVALID_PARENT        = PARENT_BIT_MASK;

    // Sanity Check
    static_assert(sizeof(uint64_t) * BYTE_BITS == (IS_LEAF_BIT_COUNT +
                                                   PARENT_BIT_COUNT +
                                                   CHILD_BIT_COUNT +
                                                   CHILD_MASK_BIT_COUNT),
                  "SVO Packed Bits exceeds 64-bit uint");

    static constexpr uint64_t   INVALID_NODE = 0x007FFFFFFFFFFFFF;
    static constexpr uint32_t   VOXEL_DIRECTION_COUNT = 8;

    // Utility Bit Options
    // Data Unpack
    static bool      IsChildrenLeaf(uint64_t packedData);
    static uint32_t  ChildMask(uint64_t packedData);
    static uint32_t  ChildrenCount(uint64_t packedData);
    static uint32_t  ChildrenIndex(uint64_t packedData);
    static uint32_t  ParentIndex(uint64_t packedData);

    static uint32_t  FindChildOffset(uint64_t packedData, uint32_t childId);
    static bool      HasChild(uint64_t packedData, uint32_t childId);

    static Vector4uc DirectionToAnisoLocations(Vector2f& interp,
                                               const Vector3f& direction);

    // Generic
    std::vector<uint32_t>       levelNodeOffsets;
    // Node Related
    std::vector<uint64_t>       nodes;
    std::vector<uint16_t>       binInfo;
    // Leaf Related
    std::vector<uint32_t>       leafParents;
    std::vector<uint16_t>       leafBinInfo;
    // Payload Related
    // Leaf
    std::vector<Vector2f>       totalIrradianceLeaf;
    std::vector<Vector2ui>      sampleCountLeaf;
    std::vector<Vector2f>       avgIrradianceLeaf;
    std::vector<uint32_t>       normalAndSpecLeaf;
    std::vector<uint8_t>        guidingFactorLeaf;
    // Node
    std::vector<Vector2f>       avgIrradianceNode;
    std::vector<uint32_t>       normalAndSpecNode;
    std::vector<uint8_t>        guidingFactorNode;

    AABB3f                      svoAABB;
    uint32_t                    voxelResolution;
    uint32_t                    leafDepth;
    uint32_t                    nodeCount;
    uint32_t                    leafCount;
    float                       leafVoxelSize;
    uint32_t                    levelOffsetCount;

    float       NodeVoxelSize(uint32_t nodeIndex, bool isLeaf) const;
    bool        Descend(uint32_t& index, uint64_t mortonCode, uint32_t levelCap) const;

    float       ConeTraceRay(bool& isLeaf, uint32_t& nodeId, const RayF&,
                             float tMin, float tMax,
                             float coneAperture = 0.0f,
                             uint32_t maxQueryLevel = 0) const;

    bool        NodeIndex(uint32_t& index, const Vector3f& worldPos,
                          uint32_t levelCap, bool checkNeighbours = false) const;

    Vector3f    ReadNormalAndSpecular(float& stdDev, float& specularity,
                                      uint32_t nodeIndex, bool isLeaf) const;

    float       ReadRadiance(const Vector3f& coneDirection, float coneAperture,
                             uint32_t nodeIndex, bool isLeaf) const;
};

class GDebugRendererSVO : public GDebugRendererI
{
    public:
        static constexpr const char* TypeName = "WFPG-SVO";

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
        static constexpr const char* SVO_TREE_NAME = "svoTrees";
        static constexpr const char* MAP_SIZE_NAME = "mapSize";
        static constexpr const char* TRACE_LEVEL_NAME = "minRayBinLevel";
        static constexpr const char* NORMALS_NAME = "normals";

        const SamplerGL         linearSampler;
        const TextureGL&        gradientTexture;
        uint32_t                curOctreeIndex;
        // All SD Trees that are loaded
        std::vector<SVOctree>   octrees;
        // Name of the Guider (shown in GUI)
        std::string             name;
        //
        bool                    multiplyCosTheta;
        Vector2ui               mapSize;
        uint32_t                minBinLevel;
        //
        TextureGL               currentTexture;
        std::vector<float>      currentValues;
        float                   maxValueDisplay;
        // Render Level Related
        Vec2uiNameList          renderResolutionNameList;
        uint32_t                renderResolutionSelectIndex;
        // Bin level related
        IntegerNameList         maxBinLevelNameList;
        uint32_t                maxBinLevelSelectIndex;
        //
        Vector2ui               normalTexSize;
        std::vector<Vector3f>   pixelNormals;

        // Shaders
        ShaderGL                compReduction;
        ShaderGL                compRefRender;

        static bool             LoadOctree(SVOctree&,
                                           const nlohmann::json& config,
                                           const std::string& configPath,
                                           uint32_t depth);

    protected:

    public:
        // Constructors & Destructor
                            GDebugRendererSVO(const nlohmann::json& config,
                                              const TextureGL& gradientTexture,
                                              const std::string& configPath,
                                              uint32_t depthCount);
                            GDebugRendererSVO(const GDebugRendererSVO&) = delete;
        GDebugRendererSVO&  operator=(const GDebugRendererSVO&) = delete;
                            ~GDebugRendererSVO();

        // Interface
        void                RenderSpatial(TextureGL&, uint32_t depth,
                                          const std::vector<Vector3f>& worldPositions) override;
        void                UpdateDirectional(const Vector3f& worldPos,
                                              const Vector2i& worldPixel,
                                              bool doLogScale,
                                              uint32_t depth) override;

        bool                RenderGUI(bool& overlayCheckboxChanged,
                                      bool& overlayValue,
                                      const ImVec2& windowSize) override;
};
