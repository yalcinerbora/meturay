#pragma once

#include "GDebugRendererI.h"

#include <nlohmann/json_fwd.hpp>
#include <GL/glew.h>

#include "RayLib/AABB.h"
#include "RayLib/Ray.h"
#include "ShaderGL.h"
#include "TextureGL.h"

struct SVOctree
{
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

    static constexpr uint32_t   LAST_BIT_UINT32 = (sizeof(uint32_t) * BYTE_BITS - 1);

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

    //
    struct AnisoRadianceF
    {
        Vector4f data[2];

        float   Read(uint8_t index) const;
        float   Read(const Vector4uc& indices,
                     const Vector2f& interp) const;
    };
    std::vector<uint64_t>       nodes;
    std::vector<AnisoRadianceF> radianceRead;
    // Leaf Related
    std::vector<uint32_t>       leafParents;
    std::vector<AnisoRadianceF> leafRadianceRead;

    AABB3f                      svoAABB;
    uint32_t                    voxelResolution;
    uint32_t                    leafDepth;
    uint32_t                    nodeCount;
    uint32_t                    leafCount;
    float                       leafVoxelSize;
    uint32_t                    levelOffsetCount;

    float       TraceRay(uint32_t& leafId, const RayF&,
                         float tMin, float tMax) const;

    bool        LeafIndex(uint32_t& index, const Vector3f& worldPos,
                          bool checkNeighbours = false) const;

    float       ReadRadiance(uint32_t nodeId, bool isLeaf,
                             const Vector3f& outgoingDir) const;
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

        const SamplerGL         linearSampler;
        const TextureGL&        gradientTexture;
        uint32_t                curOctreeIndex;
        // All SD Trees that are loaded
        std::vector<SVOctree>   octrees;
        // Name of the Guider (shown in GUI)
        std::string             name;
        //
        Vector2ui               mapSize;
        //
        TextureGL               currentTexture;
        std::vector<float>      currentValues;
        float                   maxValueDisplay;
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
                                              bool doLogScale,
                                              uint32_t depth) override;

        bool                RenderGUI(bool& overlayCheckboxChanged,
                                      bool& overlayValue,
                                      const ImVec2& windowSize) override;
};
