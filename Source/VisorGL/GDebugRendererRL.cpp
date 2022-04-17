#include "GDebugRendererRL.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <execution>
#include <Imgui/imgui.h>

#include "RayLib/FileSystemUtility.h"
#include "RayLib/Log.h"
#include "RayLib/RandomColor.h"

#include "TextureGL.h"
#include "GuideDebugStructs.h"
#include "GuideDebugGUIFuncs.h"
#include "GLConversionFunctions.h"

inline float DistanceFunction(const Vector3f& worldPos,
                              const SurfaceLeaf& leaf)
{
    return (worldPos - leaf.position).Length();
}

uint32_t SurfaceLBVH::FindNearestPoint(float& distance, const Vector3f& worldPos) const
{
    const SurfaceLeaf worldSurface = SurfaceLeaf{worldPos, Zero3f, UINT32_MAX};

    // Distance initialization function
    auto InitClosestDistance = [&](const SurfaceLeaf& worldSurface) -> float
    {
        auto DetermineDistance = [&](const SurfaceLBVHNode* childNode)
        {
            float childDistance = FLT_MAX;
            if(childNode->isLeaf)
            {
                childDistance = DistanceFunction(childNode->leaf.position, worldSurface);
            }
            else
            {
                AABB3f aabbChild = AABB3f(childNode->body.aabbMin,
                                          childNode->body.aabbMax);
                if(aabbChild.IsInside(worldSurface.position))
                {
                    childDistance = 0.0f;
                }
                else
                {
                    childDistance = aabbChild.FurthestCorner(worldSurface.position).Length()
                        + MathConstants::Epsilon;
                }
            }
            return childDistance;
        };

        // Utilize BVH as a Kd Tree and do AABB-Point
        // intersection instead of AABB-Sphere.
        // This is not an exact solution and may fail
        const SurfaceLBVHNode* currentNode = nodes.data() + rootIndex;
        // Descent towards the tree
        while(!currentNode->isLeaf)
        {
            // Check both child here
            const SurfaceLBVHNode* leftNode = nodes.data() + currentNode->body.left;
            const SurfaceLBVHNode* rightNode = nodes.data() + currentNode->body.right;
            // Determine the distances
            float leftDistance = DetermineDistance(leftNode);
            float rightDistance = DetermineDistance(rightNode);
            // Select the closest child
            currentNode = (leftDistance < rightDistance) ? leftNode : rightNode;
        }
        // We found a leaf so use it
        return DistanceFunction(currentNode->leaf.position, worldSurface) + MathConstants::Epsilon;

    };

    // Stack functions for traversal
    static constexpr uint32_t MAX_DEPTH = 64;
        // Minimal stack to traverse
    uint32_t sLocationStack[MAX_DEPTH];
    // Convenience Functions
    auto Push = [&sLocationStack](uint8_t& depth, uint32_t loc) -> void
    {
        uint32_t index = depth;
        sLocationStack[index] = loc;
        depth++;
    };
    auto ReadTop = [&sLocationStack](uint8_t depth) -> uint32_t
    {
        uint32_t index = depth;
        return sLocationStack[index];
    };
    auto Pop = [&ReadTop](uint8_t& depth) -> uint32_t
    {
        depth--;
        return ReadTop(depth);
    };

    // Initialize with an approximate closest value
    float closestDistance = InitClosestDistance(worldSurface);
    uint32_t closestLeafIndex = UINT32_MAX;
    // TODO: There is an optimization here
    // first iteration until leaf is always true
    // initialize closest distance with the radius
    uint8_t depth = 0;
    Push(depth, rootIndex);
    const SurfaceLBVHNode* currentNode = nullptr;
    while(depth > 0)
    {
        uint32_t loc = Pop(depth);
        currentNode = nodes.data() + loc;

        if(currentNode->isLeaf)
        {
            float distance = DistanceFunction(currentNode->leaf.position, worldSurface);
            if(distance < closestDistance)
            {
                closestDistance = distance;
                closestLeafIndex = currentNode->leaf.leafId;
            }
        }
        else if(AABB3f aabb = AABB3f(currentNode->body.aabbMin,
                                     currentNode->body.aabbMax);
                aabb.IntersectsSphere(worldSurface.position,
                                      closestDistance))
        {
            // Push to stack
            Push(depth, currentNode->body.right);
            Push(depth, currentNode->body.left);
        }
    }
    distance = closestDistance;
    return closestLeafIndex;
}

float SurfaceLBVH::VoronoiCenterSize() const
{
    const AABB3f sceneAABB(nodes[rootIndex].body.aabbMin,
                           nodes[rootIndex].body.aabbMax);
    Vector3f span = sceneAABB.Span();
    float sceneSize = span.Length();
    static constexpr float VORONOI_RATIO = 1.0f / 1'300.0f;
    return sceneSize * VORONOI_RATIO;
}

bool GDebugRendererRL::LoadLBVH(SurfaceLBVH& bvh,
                                const nlohmann::json& config,
                                const std::string& configPath)
{
    auto loc = config.find(LBVH_NAME);
    if(loc == config.end()) return false;

    std::string fileName = (*loc);
    std::string fileMergedPath = Utility::MergeFileFolder(configPath, fileName);
    std::ifstream file(fileMergedPath, std::ios::binary);
    if(!file.good()) return false;
    // Assume both architectures are the same (writer, reader)
    static_assert(sizeof(char) == sizeof(Byte), "\"Byte\" is not have sizeof(char)");
    // Read STree Start Offset
    file.read(reinterpret_cast<char*>(&bvh.rootIndex), sizeof(uint32_t));
    // Read STree Node Count
    file.read(reinterpret_cast<char*>(&bvh.nodeCount), sizeof(uint32_t));
    // Read DTree Count
    file.read(reinterpret_cast<char*>(&bvh.leafCount), sizeof(uint32_t));
    // Read DTree Offset/Count Pairs
    bvh.nodes.resize(bvh.nodeCount);
    file.read(reinterpret_cast<char*>(bvh.nodes.data()),
              sizeof(SurfaceLBVHNode) * bvh.nodeCount);
    assert(bvh.nodes.size() == bvh.nodeCount);
    return true;
}

GDebugRendererRL::GDebugRendererRL(const nlohmann::json& config,
                                   const TextureGL& gradientTexture,
                                   const std::string& configPath,
                                   uint32_t depthCount)
    : linearSampler(SamplerGLEdgeResolveType::CLAMP,
                    SamplerGLInterpType::LINEAR)
    , gradientTexture(gradientTexture)
    , compReduction(ShaderType::COMPUTE, u8"Shaders/TextureMaxReduction.comp")
    , compRefRender(ShaderType::COMPUTE, u8"Shaders/PGReferenceRender.comp")
    , renderPerimeter(false)
{
    if(!LoadLBVH(lbvh, config, configPath))
        throw std::runtime_error("Unable to Load LBVH");
    // Load the Name
    name = config[GuideDebug::NAME];

    // Load QFunctions from the files as well
    for(std::string fName : config[QFUNC_NAME])
    {
        qFuncFileNames.push_back(Utility::MergeFileFolder(configPath, fName));
    }
    // Load QFunc Size
    qFuncSize = SceneIO::LoadVector<2, uint32_t>(config[QSIZE_NAME]);
    // Allocate the texture for rendering
    currentTexture = TextureGL(qFuncSize, PixelFormat::RGBA8_UNORM);
    // Allocate values array even if it is empty
    currentValues.resize(qFuncSize.Multiply());
}

GDebugRendererRL::~GDebugRendererRL()
{

}

void GDebugRendererRL::RenderSpatial(TextureGL& overlayTex, uint32_t,
                                     const std::vector<Vector3f>& worldPositions)
{
    // Parallel Transform the world pos to color
    std::vector<Byte> pixelColors;
    pixelColors.resize(worldPositions.size() * sizeof(Vector3f));
    std::transform(std::execution::par_unseq,
                   worldPositions.cbegin(), worldPositions.cend(),
                   reinterpret_cast<Vector3f*>(pixelColors.data()),
                   [&](const Vector3f& pos)
                   {
                       float distance;
                       uint32_t index = lbvh.FindNearestPoint(distance, pos);
                       Vector3f locColor = (distance <= lbvh.VoronoiCenterSize())
                                           ? Zero3f
                                           : Utility::RandomColorRGB(index);
                       return locColor;
                   });

    // Copy Transform to the overlay texture
    overlayTex.CopyToImage(pixelColors, Vector2ui(0),
                           overlayTex.Size(),
                           PixelFormat::RGB_FLOAT);
}

void GDebugRendererRL::UpdateDirectional(const Vector3f& worldPos,
                                         bool doLogScale,
                                         uint32_t depth)
{
    const std::string& fileName = qFuncFileNames[depth];


    // Query leaf
    float distance;
    uint32_t leafIndex = lbvh.FindNearestPoint(distance, worldPos);

    size_t dirPortionByteCount = qFuncSize.Multiply() * sizeof(float);
    size_t fileStart = leafIndex * dirPortionByteCount;

    // Load from the qFile of the requested depth
    std::ifstream file(fileName, std::ios_base::binary);
    // Read only the required portion
    file.seekg(fileStart);
    std::vector<Byte> data(dirPortionByteCount);
    file.read(reinterpret_cast<char*>(data.data()), dirPortionByteCount);

    // Copy the actual current values to the member variable as well.
    currentValues.resize(qFuncSize.Multiply());
    std::memcpy(currentValues.data(), data.data(), dirPortionByteCount);

    // Load temporarily to a texture
    TextureGL qTexture = TextureGL(qFuncSize, PixelFormat::R_FLOAT);
    qTexture.CopyToImage(data, Zero2ui, qFuncSize, PixelFormat::R_FLOAT);

    // ============================= //
    //     Call Reduction Shader     //
    // ============================= //
    // Get a max luminance buffer;
    float initalMaxData = 0.0f;
    GLuint maxBuffer;
    glGenBuffers(1, &maxBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, maxBuffer);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, 1 * sizeof(float), &initalMaxData, 0);

    // Both of these compute shaders total work count is same
    const GLuint workCount = qTexture.Size()[1] * qTexture.Size()[0];
    // Some WG Definitions (statically defined in shader)
    static constexpr GLuint WORK_GROUP_1D_X = 256;
    static constexpr GLuint WORK_GROUP_2D_X = 16;
    static constexpr GLuint WORK_GROUP_2D_Y = 16;
    // =======================================================
    // Set Max Shader
    compReduction.Bind();
    // Bind Uniforms
    glUniform2ui(U_RES, qTexture.Size()[0], qTexture.Size()[1]);
    // Bind SSBO
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSB_MAX_LUM, maxBuffer);
    // Textures
    qTexture.Bind(T_IN_LUM_TEX);
    // Dispatch Max Shader
    // Max shader is 1D shader set data accordingly
    GLuint gridX_1D = (workCount + WORK_GROUP_1D_X - 1) / WORK_GROUP_1D_X;
    glDispatchCompute(gridX_1D, 1, 1);
    glMemoryBarrier(GL_UNIFORM_BARRIER_BIT |
                    GL_SHADER_STORAGE_BARRIER_BIT);
    // =======================================================
    // Unbind SSBO just to be sure
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSB_MAX_LUM, 0);
    // ============================= //
    //     Call Reduction Shader     //
    // ============================= //
     // Set Render Shader
    compRefRender.Bind();
    // Bind Uniforms
    glUniform2ui(U_RES, qTexture.Size()[0], qTexture.Size()[1]);
    glUniform1i(U_LOG_ON, doLogScale ? 1 : 0);
    //
    // UBOs
    glBindBufferBase(GL_UNIFORM_BUFFER, UB_MAX_LUM, maxBuffer);
    // Textures
    qTexture.Bind(T_IN_LUM_TEX);
    gradientTexture.Bind(T_IN_GRAD_TEX);  linearSampler.Bind(T_IN_GRAD_TEX);
    // Images
    glBindImageTexture(I_OUT_REF_IMAGE, currentTexture.TexId(),
                       0, false, 0, GL_WRITE_ONLY,
                       PixelFormatToSizedGL(currentTexture.Format()));
    // Dispatch Render Shader
    // Max shader is 2D shader set data accordingly
    GLuint gridX_2D = (qTexture.Size()[0] + WORK_GROUP_2D_X - 1) / WORK_GROUP_2D_X;
    GLuint gridY_2D = (qTexture.Size()[1] + WORK_GROUP_2D_Y - 1) / WORK_GROUP_2D_Y;
    glDispatchCompute(gridX_2D, gridY_2D, 1);
    // =======================================================
    // All done!!!

    // Delete Temp Max Buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glDeleteBuffers(1, &maxBuffer);
}

bool GDebugRendererRL::RenderGUI(bool& overlayCheckboxChanged,
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

        ImGui::Text("Max Value: %f", maxValueDisplay);
        ImGui::Text("Location Id : %u", curLocationIndex);
        ImGui::EndPopup();

    }
    ImGui::EndChild();
    return changed;
}