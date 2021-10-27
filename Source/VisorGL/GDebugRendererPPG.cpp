#include "GDebugRendererPPG.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <execution>
#include <atomic>

#include "RayLib/FileSystemUtility.h"
#include "RayLib/Log.h"

#include "TextureGL.h"
#include "GuideDebugStructs.h"


static const uint8_t QUAD_INDICES[6] = { 0, 1, 2, 0, 2, 3};
static const float QUAD_VERTEX_POS[4 * 3] =
{
    0, 0,
    1, 0,
    1, 1,
    0, 1
};

GDebugRendererPPG::GDebugRendererPPG(const nlohmann::json& config,
                                     const TextureGL& gradientTexture,
                                     const std::string& configPath,
                                     uint32_t depthCount)
    : gradientTexture(gradientTexture)
    , configPath(configPath)
    , depthCount(depthCount)
    , fbo(0)
    , vao(0)
    , vPos(0)
    , indexBuffer(0)
    , vPosBuffer(0)
    , treeBuffer(0)
    , treeBufferSize(0)
    , perimeterColor(1.0f, 1.0f, 1.0f)
    , vertDTreeRender(ShaderType::VERTEX, u8"Shaders/DTreeRender.vert")
    , fragDTreeRender(ShaderType::FRAGMENT, u8"Shaders/DTreeRender.frag")
    , linearSampler(SamplerGLEdgeResolveType::CLAMP,
                    SamplerGLInterpType::LINEAR)
{
    glGenFramebuffers(1, &fbo);

    // Create Static Buffers
    glGenBuffers(1, &vPosBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vPosBuffer);
    glBufferStorage(GL_ARRAY_BUFFER, 4 * 2 * sizeof(float), QUAD_VERTEX_POS, 0);

    glGenBuffers(1, &indexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, indexBuffer);
    glBufferStorage(GL_ARRAY_BUFFER, 6 * sizeof(uint8_t), QUAD_INDICES, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Create your VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    // Vertex Position
    constexpr GLenum VPOS_ATTRIB = 0;
    glEnableVertexAttribArray(IN_POS);
    glVertexAttribFormat(IN_POS, 2, GL_FLOAT, false, 0);
    glVertexAttribBinding(IN_POS, IN_POS);
    glBindVertexBuffer(IN_POS, vPosBuffer, 0, sizeof(float) * 2);
    // Vertex Indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    // Per-Instance Related
    glEnableVertexAttribArray(IN_OFFSET);
    glVertexAttribFormat(IN_OFFSET, 2, GL_FLOAT, false, 0);
    glVertexAttribBinding(IN_OFFSET, IN_OFFSET);
    glVertexAttribDivisor(IN_OFFSET, 1);

    glEnableVertexAttribArray(IN_DEPTH);
    glVertexAttribIFormat(IN_DEPTH, 1, GL_UNSIGNED_INT, 0);
    glVertexAttribBinding(IN_DEPTH, IN_DEPTH);
    glVertexAttribDivisor(IN_DEPTH, 1);

    glEnableVertexAttribArray(IN_RADIANCE);
    glVertexAttribFormat(IN_RADIANCE, 1, GL_FLOAT, false, 0);
    glVertexAttribBinding(IN_RADIANCE, IN_RADIANCE);
    glVertexAttribDivisor(IN_RADIANCE, 1);

    glBindVertexArray(0);

    // Load the Name
    name = config[GuideDebug::NAME];
    // Load SDTrees to memory
    sdTrees.resize(depthCount);
    for(uint32_t i = 0; i < depthCount; i++)
    {
        LoadSDTree(sdTrees[i], config, configPath, i);
    }
    // All done!
}

GDebugRendererPPG::~GDebugRendererPPG()
{
    glDeleteFramebuffers(1, &fbo);
}

bool GDebugRendererPPG::LoadSDTree(SDTree& sdTree,
                                   const nlohmann::json& config,
                                   const std::string& configPath,
                                   uint32_t depth)
{
    auto loc = config.find(SD_TREE_NAME);
    if(loc == config.end()) return false;
    if(depth >= loc->size()) return false;

    std::string fileName = (*loc)[depth];
    std::string fileMergedPath = Utility::MergeFileFolder(configPath, fileName);
    std::ifstream file(fileMergedPath, std::ios::binary);
    std::istreambuf_iterator<char>fileIt(file);
    static_assert(sizeof(char) == sizeof(Byte), "\"Byte\" is not have sizeof(char)");
    // Read STree Start Offset
    uint64_t sTreeOffset;
    file.read(reinterpret_cast<char*>(&sTreeOffset), sizeof(uint64_t));
    // Read STree Node Count
    uint64_t sTreeNodeCount;
    file.read(reinterpret_cast<char*>(&sTreeNodeCount), sizeof(uint64_t));
    // Read DTree Count
    uint64_t dTreeCount;
    file.read(reinterpret_cast<char*>(&dTreeCount), sizeof(uint64_t));
    // Read DTree Offset/Count Pairs
    std::vector<Vector2ul> offsetCountPairs(dTreeCount);
    file.read(reinterpret_cast<char*>(offsetCountPairs.data()), sizeof(Vector2ul) * dTreeCount);
    // Read STree
    // Extents
    file.read(reinterpret_cast<char*>(&sdTree.extents), sizeof(AABB3f));
    // Nodes
    sdTree.sTreeNodes.resize(sTreeNodeCount);
    file.read(reinterpret_cast<char*>(sdTree.sTreeNodes.data()),
              sizeof(STreeNode) * sTreeNodeCount);
    // Read DTrees in order
    for(uint64_t i = 0; i < dTreeCount; i++)
    {
        size_t fileOffset = offsetCountPairs[i][0];
        size_t nodeCount = offsetCountPairs[i][1];

        file.seekg(fileOffset);
        // Read Base
        std::pair<uint32_t, float> dTreeBase;
        file.read(reinterpret_cast<char*>(&dTreeBase.first), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&dTreeBase.second), sizeof(float));
        // Read Nodes
        std::vector<DTreeNode> dTreeNodes(nodeCount);
        file.read(reinterpret_cast<char*>(dTreeNodes.data()), nodeCount * sizeof(DTreeNode));
        // Move to the struct
        sdTree.dTrees.push_back(std::move(dTreeBase));
        sdTree.dTreeNodes.push_back(std::move(dTreeNodes));
    }
    return true;
}

void GDebugRendererPPG::RenderSpatial(TextureGL&, uint32_t depth)
{
    // TODO:
}

void GDebugRendererPPG::RenderDirectional(TextureGL& tex,
                                          std::vector<float>& values,
                                          const Vector3f& worldPos,
                                          bool doLogScale,
                                          uint32_t depth)
{
    // Find DTree
    const SDTree& currentSDTree = sdTrees[depth];
    uint32_t dTreeIndex = currentSDTree.FindDTree(worldPos);
    const auto& dTreeNodes = currentSDTree.dTreeNodes[dTreeIndex];
    const auto& dTreeValues = currentSDTree.dTrees[dTreeIndex];
    // Find out leaf count (a.k.a square count)
    std::atomic_size_t squareCount = 0;
    if(dTreeNodes.size() == 0)
        squareCount = 1;
    else
        std::for_each(std::execution::par_unseq, dTreeNodes.cbegin(), dTreeNodes.cend(),
                      [&squareCount] (const DTreeNode& node)
                      {
                          if(node.IsLeaf(0)) squareCount++;
                          if(node.IsLeaf(1)) squareCount++;
                          if(node.IsLeaf(2)) squareCount++;
                          if(node.IsLeaf(3)) squareCount++;
                      });

    // Compile DTree Data for GPU
    static_assert(sizeof(Vector2f) == (sizeof(float) * 2), "Vector2f != sizeof(float) * 2");
    size_t newTreeSize = squareCount * (sizeof(float) * 3 + sizeof(uint32_t));
    std::vector<Byte> treeBufferCPU(newTreeSize);
    size_t offset = 0;
    Vector2f* offsetStart = reinterpret_cast<Vector2f*>(treeBufferCPU.data() + offset);
    offset += squareCount * sizeof(Vector2f);
    uint32_t* depthStart = reinterpret_cast<uint32_t*>(treeBufferCPU.data() + offset);
    offset += squareCount * sizeof(uint32_t);
    float* radianceStart = reinterpret_cast<float*>(treeBufferCPU.data() + offset);
    offset += squareCount * sizeof(float);
    assert(newTreeSize == offset);
    // Generate GPU Data
    std::atomic<float> maxRadiance = -std::numeric_limits<float>::max();
    std::atomic_uint32_t maxDepth = 0;
    std::atomic_uint32_t allocator = 0;
    auto CalculateGPUData = [&] (const DTreeNode& node)
    {
        for(uint8_t i = 0; i < 4; i++)
        {
            if(!node.IsLeaf(i)) continue;
            // Allocate an index
            uint32_t location = allocator++;
            // Calculate Irrad max irrad etc.
            float irrad = node.irradianceEstimates[i];
            // Calculate Depth & Offset
            uint32_t depth = 1;
            Vector2f offset(((i >> 0) & 0b01) ? 0.5f : 0.0f,
                            ((i >> 1) & 0b01) ? 0.5f : 0.0f);

            // Leaf -> Root Traverse
            const DTreeNode* curNode = &node;
            while(!curNode->IsRoot())
            {
                const DTreeNode* parentNode = &dTreeNodes[curNode->parentIndex];
                uint32_t nodeIndex = static_cast<uint32_t>(curNode - dTreeNodes.data());
                // Determine which child are you
                uint32_t childId = UINT32_MAX;
                childId = (parentNode->childIndices[0] == nodeIndex) ? 0 : childId;
                childId = (parentNode->childIndices[1] == nodeIndex) ? 1 : childId;
                childId = (parentNode->childIndices[2] == nodeIndex) ? 2 : childId;
                childId = (parentNode->childIndices[3] == nodeIndex) ? 3 : childId;
                // Calculate your offset
                Vector2f childCoordOffset(((childId >> 0) & 0b01) ? 0.5f : 0.0f,
                                          ((childId >> 1) & 0b01) ? 0.5f : 0.0f);
                offset = childCoordOffset + 0.5f * offset;
                depth++;
                // Traverse upwards
                curNode = parentNode;
            }

            // Atomic MAX DEPTH
            uint32_t expectedDepth = maxDepth.load();
            while(!maxDepth.compare_exchange_strong(expectedDepth,
                                                    std::max(expectedDepth, depth)));
            // Store
            radianceStart[location] = irrad;
            depthStart[location] = depth;
            offsetStart[location] = offset;
        }
    };
    auto CalculateMaxIrrad = [&] (uint32_t index)
    {
        float irrad = radianceStart[index];
        uint32_t depth = depthStart[index];

        // Normalize irrad using depth/maxDept;
        irrad /= static_cast<float>(1 << (2 * (maxDepth - depth)));

        // Atomic MAX IRRAD
        float expectedIrrad = maxRadiance.load();
        while(!maxRadiance.compare_exchange_strong(expectedIrrad,
                                                   std::max(expectedIrrad, irrad)));
    };

    // Edge case of node is parent and leaf
    if(dTreeNodes.size() == 0)
    {
        depthStart[0] = 0;
        offsetStart[0] = Zero2f;

        maxRadiance =  dTreeValues.second;
        radianceStart[0] = dTreeValues.second;
    }
    else
    {
        std::for_each(std::execution::par_unseq,
                      dTreeNodes.cbegin(),
                      dTreeNodes.cend(),
                      CalculateGPUData);
        // After that calculate max irradiance
        // TODO: use ranges (c++20) when available
        //std::vector<uint32_t> indices(squareCount);
        //std::iota(indices.begin(), indices.end(), 0);
        //std::for_each(std::execution::par_unseq,
        //              indices.cbegin(),
        //              indices.cend(),
        //              CalculateMaxIrrad);
        maxRadiance = std::reduce(std::execution::par_unseq,
                                  radianceStart, radianceStart + squareCount,
                                  -std::numeric_limits<float>::max(),
                                  [](const float a, const float b)->float
                                  {
                                      return std::max(a, b);
                                  });

        // Check that we properly did all
        assert(allocator.load() == squareCount.load());
    }

    METU_LOG("Max Rad {:f}", maxRadiance.load());

    // Gen Temp Texture for Value rendering
    TextureGL valueTex(tex.Size(), PixelFormat::R_FLOAT);

    // Generate/Resize Buffer
    if(treeBufferSize < newTreeSize)
    {
        glDeleteBuffers(1, &treeBuffer);
        glGenBuffers(1, &treeBuffer);
        glBindBuffer(GL_ARRAY_BUFFER, treeBuffer);
        glBufferStorage(GL_ARRAY_BUFFER, newTreeSize, nullptr, GL_DYNAMIC_STORAGE_BIT);
        treeBufferSize = newTreeSize;
    }
    // Load Buffers
    glBindBuffer(GL_ARRAY_BUFFER, treeBuffer);
    glBufferSubData(GL_ARRAY_BUFFER, 0, newTreeSize, treeBufferCPU.data());
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Bind Buffers
    size_t depthOffset = squareCount * sizeof(float) * 2;
    size_t radianceOffset = squareCount * (sizeof(float) * 2 + sizeof(uint32_t));
    glBindVertexArray(vao);
    glBindVertexBuffer(IN_OFFSET, treeBuffer, 0, sizeof(float) * 2);
    glBindVertexBuffer(IN_DEPTH, treeBuffer, depthOffset, sizeof(uint32_t));
    glBindVertexBuffer(IN_RADIANCE, treeBuffer, radianceOffset, sizeof(float));

    // Bind FBO
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0 + OUT_COLOR,
                           GL_TEXTURE_2D, tex.TexId(), 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0 + OUT_VALUE,
                           GL_TEXTURE_2D, valueTex.TexId(), 0);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    // Enable 2 Color Channels
    const GLenum Attachments[2] = {GL_COLOR_ATTACHMENT0 + OUT_COLOR,
                                   GL_COLOR_ATTACHMENT0 + OUT_VALUE};
    glDrawBuffers(2, Attachments);

    // Change Viewport
    glViewport(0, 0, tex.Width(), tex.Height());

    // Global States
    glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
    //glDisable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT);

    // ==========================//
    //   Render Filled Squares   //
    // ==========================//
    // Bind Texture
    gradientTexture.Bind(T_IN_GRADIENT);
    linearSampler.Bind(T_IN_GRADIENT);
    // Bind V Shader
    vertDTreeRender.Bind();
    // Uniforms
    glUniform1f(U_MAX_RADIANCE, maxRadiance);
    glUniform1ui(U_MAX_DEPTH, maxDepth);
    glUniform1i(U_LOG_ON, doLogScale ? 1 : 0);
    // Bind F Shader
    fragDTreeRender.Bind();
    // Uniforms
    glUniform1i(U_PERIMIETER_ON, 0);
    // Draw Call
    glDrawElementsInstanced(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, nullptr,
                            static_cast<GLsizei>(squareCount));
    //=================//
    //   Render Lines  //
    //=================//
    // Same thing but only push a different uniforms and draw call
    // Bind Uniforms (Frag Shader is Already Bound)
    glUniform1i(U_PERIMIETER_ON, 1);
    glUniform3f(U_PERIMIETER_COLOR, perimeterColor[0], perimeterColor[1], perimeterColor[2]);
    // Set Line Width
    glEnable(GL_LINE_SMOOTH);
    glLineWidth(3.0f);
    // Draw Call
    glDrawArraysInstanced(GL_LINE_LOOP, 0, 4, static_cast<GLsizei>(squareCount));

    // Rebind the window framebuffer etc..
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Get Value Buffer to CPU
    values.resize(tex.Size()[0] * tex.Size()[1]);
    glBindTexture(GL_TEXTURE_2D, valueTex.TexId());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT,
                  values.data());

    // All Done!
}

const std::string& GDebugRendererPPG::Name() const
{
    return name;
}