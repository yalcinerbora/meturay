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
    // Resulting Closest Leaf Index
    // & Closest Hit
    // Arbitrarily set the initial distance 
    // to the first leaf(node[0]) distance
    assert(nodes[0].isLeaf == true);
    float closestDistance = DistanceFunction(worldPos,
                                             nodes[0].leaf);
    uint32_t closestIndex = UINT32_MAX;
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
            float distance = DistanceFunction(worldPos, currentNode->leaf);
            if(distance < closestDistance)
            {
                closestDistance = distance;
                closestIndex = static_cast<uint32_t>(currentNode - nodes.data());
            }
        }
        else if(AABB3f aabb = AABB3f(currentNode->body.aabbMin,
                                     currentNode->body.aabbMax);
                aabb.IntersectsSphere(worldPos, closestDistance))
        {
            // Push to stack
            Push(depth, currentNode->body.right);
            Push(depth, currentNode->body.left);
        }
    }
    distance = closestDistance;
    return closestIndex;
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
    // Assume both architechtures are the same (writer, reader)
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
    , renderPerimeter(false)
{
    if(!LoadLBVH(lbvh, config, configPath))
        throw std::runtime_error("Unable to Load LBVH");
    // Load the Name
    name = config[GuideDebug::NAME];

    // Load QFunctions from the files aswell
    // TODO:
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
    // TODO:
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