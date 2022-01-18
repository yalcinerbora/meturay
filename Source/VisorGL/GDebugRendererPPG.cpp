#include "GDebugRendererPPG.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <execution>
#include <atomic>
#include <Imgui/imgui.h>

#include "RayLib/FileSystemUtility.h"
#include "RayLib/Log.h"
#include "RayLib/RandomColor.h"

#include "TextureGL.h"
#include "GuideDebugStructs.h"
#include "GuideDebugGUIFuncs.h"
#include "GLConversionFunctions.h"


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
    : linearSampler(SamplerGLEdgeResolveType::CLAMP,
                    SamplerGLInterpType::LINEAR)
    , gradientTexture(gradientTexture)
    , perimeterColor(1.0f, 1.0f, 1.0f)
    , currentTexture(GuideDebugGUIFuncs::PG_TEXTURE_SIZE, PixelFormat::RGB8_UNORM)
    , currentValues(GuideDebugGUIFuncs::PG_TEXTURE_SIZE[0] * GuideDebugGUIFuncs::PG_TEXTURE_SIZE[1], 0.0f)
    , maxValueDisplay(0.0f)
    , renderPerimeter(false)
    , renderSamples(false)
    , fbo(0)
    , vao(0)
    , indexBuffer(0)
    , vPosBuffer(0)
    , treeBuffer(0)
    , treeBufferSize(0)
    , vertDTreeRender(ShaderType::VERTEX, u8"Shaders/DTreeRender.vert")
    , fragDTreeRender(ShaderType::FRAGMENT, u8"Shaders/DTreeRender.frag")
    , compSTreeRender(ShaderType::COMPUTE, u8"Shaders/STreeRender.comp")
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
    if(!file.good()) return false;

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
        std::vector<DTreeNodeCPU> dTreeNodes(nodeCount);
        file.read(reinterpret_cast<char*>(dTreeNodes.data()), nodeCount * sizeof(DTreeNodeCPU));
        // Move to the struct
        sdTree.dTrees.push_back(std::move(dTreeBase));
        sdTree.dTreeNodes.push_back(std::move(dTreeNodes));
    }
    return true;
}

void GDebugRendererPPG::RenderSpatial(TextureGL& overlayTex, uint32_t depth,
                                      const std::vector<Vector3f>& worldPositions)
{
    // GLSL compatible node data
    struct STreeNodeSSBO
    {
        int32_t axisType;
        uint32_t index;
        int32_t isLeaf;
    };

    const SDTree& sdTree = sdTrees[depth];
    std::vector<Vector4f> colors(sdTree.dTreeNodes.size());
    std::vector<Vector4f> expWorldPositions(worldPositions.size());
    std::vector<STreeNodeSSBO> sTreeNodesGL(sdTree.sTreeNodes.size());

    Vector2ui resolution = overlayTex.Size();
    assert(worldPositions.size() == resolution[0] * resolution[1]);

    int colorId = 0;
    for(size_t i = 0; i < worldPositions.size(); i++)
    {
        // Expand the vec3 to vec4
        // (std430 SSBO requires vec4 alignment for vec3's)
        expWorldPositions[i] = Vector4f(worldPositions[i], 0.0f);
    }
    for(size_t i = 0; i < sdTree.sTreeNodes.size(); i++)
    {
        // Generate glsl capable sTree node
        sTreeNodesGL[i].axisType = static_cast<int>(sdTree.sTreeNodes[i].splitAxis);
        sTreeNodesGL[i].index = sdTree.sTreeNodes[i].index;
        sTreeNodesGL[i].isLeaf = sdTree.sTreeNodes[i].isLeaf ? 1 : 0;
        // Determine leaf color
        using Utility::RandomColorRGB;
        if(sdTree.sTreeNodes[i].isLeaf)
        {
            // Also make the color 4 channel because of std430 layout
            colors[colorId] = Vector4f(RandomColorRGB(colorId), 0.0f);
            colorId++;
        }
    }
    // Allocate buffers & send to GPU
    GLuint buffers[3];
    glGenBuffers(3, buffers);
    GLuint colorBuffer = buffers[0];
    glBindBuffer(GL_COPY_WRITE_BUFFER, colorBuffer);
    glBufferStorage(GL_COPY_WRITE_BUFFER,
                    sizeof(Vector4f) * colors.size(),
                    colors.data(), 0);
    GLuint nodeBuffer = buffers[1];
    glBindBuffer(GL_COPY_WRITE_BUFFER, nodeBuffer);
    glBufferStorage(GL_COPY_WRITE_BUFFER,
                    sizeof(STreeNodeSSBO)* sTreeNodesGL.size(),
                    sTreeNodesGL.data(), 0);
    GLuint worldPosBuffer = buffers[2];
    glBindBuffer(GL_COPY_WRITE_BUFFER, worldPosBuffer);
    glBufferStorage(GL_COPY_WRITE_BUFFER,
                    sizeof(Vector4f)* expWorldPositions.size(),
                    expWorldPositions.data(), 0);

    compSTreeRender.Bind();
    // Get Ready to call shader
    // Bindings
    // SSBOs
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSB_LEAF_COL,
                     colorBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSB_STREE,
                     nodeBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSB_WORLD_POS,
                     worldPosBuffer);
    // Uniforms
    glUniform2i(U_RES, static_cast<int>(resolution[0]),
                       static_cast<int>(resolution[1]));
    glUniform3f(U_AABB_MIN, sdTree.extents.Min()[0],
                sdTree.extents.Min()[1], sdTree.extents.Min()[2]);
    glUniform3f(U_AABB_MAX, sdTree.extents.Max()[0],
                sdTree.extents.Max()[1], sdTree.extents.Max()[2]);
    glUniform1ui(U_NODE_COUNT,
                 static_cast<uint32_t>(sdTree.sTreeNodes.size()));
    // Images
    glBindImageTexture(I_OUT_IMAGE, overlayTex.TexId(),
                       0, false, 0, GL_WRITE_ONLY,
                       PixelFormatToSizedGL(overlayTex.Format()));

    // Call the Kernel
    GLuint gridX = (resolution[0] + 16 - 1) / 16;
    GLuint gridY = (resolution[1] + 16 - 1) / 16;
    glDispatchCompute(gridX, gridY, 1);
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                    GL_TEXTURE_FETCH_BARRIER_BIT);
}

void GDebugRendererPPG::UpdateDirectional(const Vector3f& worldPos,
                                          bool doLogScale,
                                          uint32_t depth)
{
    // Find DTree
    const SDTree& currentSDTree = sdTrees[depth];
    curDTreeIndex = currentSDTree.FindDTree(worldPos);
    const auto& dTreeNodes = currentSDTree.dTreeNodes[curDTreeIndex];
    const auto& dTreeValues = currentSDTree.dTrees[curDTreeIndex];
    // Find out leaf count (a.k.a square count)
    std::atomic_size_t squareCount = 0;
    if(dTreeNodes.size() == 0)
        squareCount = 1;
    else
        std::for_each(std::execution::par_unseq, dTreeNodes.cbegin(), dTreeNodes.cend(),
                      [&squareCount] (const DTreeNodeCPU& node)
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
    float* valueStart = reinterpret_cast<float*>(treeBufferCPU.data() + offset);
    offset += squareCount * sizeof(float);
    assert(newTreeSize == offset);
    // Generate GPU Data
    std::atomic<float> maxValue = -std::numeric_limits<float>::max();
    std::atomic_uint32_t maxDepth = 0;
    std::atomic_uint32_t allocator = 0;
    auto CalculateGPUData = [&] (const DTreeNodeCPU& node)
    {
        for(uint8_t i = 0; i < 4; i++)
        {
            if(!node.IsLeaf(i)) continue;
            // Allocate an index
            uint32_t location = allocator++;
            // Calculate Irrad max irrad etc.
            float value = renderSamples
                            ? static_cast<uint32_t>(node.sampleCounts[i])
                            : node.irradianceEstimates[i];
            // Calculate Depth & Offset
            uint32_t depth = 1;
            Vector2f offset(((i >> 0) & 0b01) ? 0.5f : 0.0f,
                            ((i >> 1) & 0b01) ? 0.5f : 0.0f);

            // Leaf -> Root Traverse
            const DTreeNodeCPU* curNode = &node;
            while(!curNode->IsRoot())
            {
                const DTreeNodeCPU* parentNode = &dTreeNodes[curNode->parentIndex];
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
            valueStart[location] = value;
            depthStart[location] = depth;
            offsetStart[location] = offset;
        }
    };
    auto CalculateMaxIrrad = [&] (uint32_t index)
    {
        float irrad = valueStart[index];
        uint32_t depth = depthStart[index];

        // Normalize irrad using depth/maxDept;
        irrad /= static_cast<float>(1 << (2 * (maxDepth - depth)));

        // Atomic MAX IRRAD
        float expectedIrrad = maxValue.load();
        while(!maxValue.compare_exchange_strong(expectedIrrad,
                                                std::max(expectedIrrad, irrad)));
    };

    // Edge case of node is parent and leaf
    if(dTreeNodes.size() == 0)
    {
        depthStart[0] = 0;
        offsetStart[0] = Zero2f;

        if(renderSamples)
            maxValue = static_cast<float>(dTreeValues.first);
        else
            maxValue = dTreeValues.second;

        valueStart[0] = maxValue;

    }
    else
    {
        std::for_each(std::execution::par_unseq,
                      dTreeNodes.cbegin(),
                      dTreeNodes.cend(),
                      CalculateGPUData);
        // After that calculate max irradiance
        // TODO: use ranges (c++20) when available
        std::vector<uint32_t> indices(squareCount);
        std::iota(indices.begin(), indices.end(), 0);
        std::for_each(std::execution::par_unseq,
                      indices.cbegin(),
                      indices.cend(),
                      CalculateMaxIrrad);
        //maxValue = std::reduce(std::execution::par_unseq,
        //                          valueStart, valueStart + squareCount,
        //                          -std::numeric_limits<float>::max(),
        //                          [](const float a, const float b)->float
        //                          {
        //                              return std::max(a, b);
        //                          });

        // Check that we properly did all
        assert(allocator.load() == squareCount.load());
    }
    maxValueDisplay = maxValue;

    // Gen Temp Texture for Value rendering
    TextureGL valueTex(currentTexture.Size(), PixelFormat::R_FLOAT);

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
                           GL_TEXTURE_2D, currentTexture.TexId(), 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER,
                           GL_COLOR_ATTACHMENT0 + OUT_VALUE,
                           GL_TEXTURE_2D, valueTex.TexId(), 0);
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

    // Enable 2 Color Channels
    const GLenum Attachments[2] = {GL_COLOR_ATTACHMENT0 + OUT_COLOR,
                                   GL_COLOR_ATTACHMENT0 + OUT_VALUE};
    glDrawBuffers(2, Attachments);

    // Change View-port
    glViewport(0, 0, currentTexture.Width(), currentTexture.Height());

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
    glUniform1f(U_MAX_RADIANCE, maxValue);
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
    if(renderPerimeter)
    {
        glUniform1i(U_PERIMIETER_ON, 1);
        glUniform3f(U_PERIMIETER_COLOR, perimeterColor[0], perimeterColor[1], perimeterColor[2]);
        // Set Line Width
        glEnable(GL_LINE_SMOOTH);
        //glLineWidth(3.0f);
        // Draw Call
        glDrawArraysInstanced(GL_LINE_LOOP, 0, 4, static_cast<GLsizei>(squareCount));
    }
    // Rebind the window framebuffer etc..
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Get Value Buffer to CPU
    currentValues.resize(currentTexture.Size()[0] * currentTexture.Size()[1]);
    glBindTexture(GL_TEXTURE_2D, valueTex.TexId());
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT,
                  currentValues.data());

    // All Done!
}

bool GDebugRendererPPG::RenderGUI(bool& overlayCheckboxChanged,
                                  bool& overlayValue,
                                  const ImVec2& windowSize)
{
    bool changed = false;
    using namespace GuideDebugGUIFuncs;

    ImGui::BeginChild(("##" + name).c_str(), windowSize, false);
    ImGui::SameLine(0.0f, CenteredTextLocation(name.c_str(), windowSize.x));
    overlayCheckboxChanged = ImGui::Checkbox("##OverlayCheckbox", &overlayValue);
    ImGui::SameLine();
    ImGui::Text("%s", name.c_str());
    ImVec2 remainingSize = FindRemainingSize(windowSize);
    remainingSize.x = remainingSize.y;
    ImGui::NewLine();
    ImGui::SameLine(0.0f, (windowSize.x - remainingSize.x) * 0.5f - ImGui::GetStyle().WindowPadding.x);
    RenderImageWithZoomTooltip(currentTexture, currentValues, remainingSize);

    if(ImGui::BeginPopupContextItem(("texPopup" + name).c_str()))
    {
        changed |= ImGui::Checkbox("RenderGrid", &renderPerimeter);
        changed |= ImGui::Checkbox("RenderSamples", &renderSamples);

        ImGui::Text("Max Value: %f", maxValueDisplay);
        ImGui::Text("DTree Id : %u", curDTreeIndex);
        ImGui::EndPopup();

    }
    ImGui::EndChild();
    return changed;
}