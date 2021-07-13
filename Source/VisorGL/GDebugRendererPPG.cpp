#include "GDebugRendererPPG.h"

#include <nlohmann/json.hpp>
#include <fstream>

#include "RayLib/FileSystemUtility.h"

GDebugRendererPPG::GDebugRendererPPG(const nlohmann::json& config,                                     
                                     const TextureGL& gradientTexture,
                                     const std::string& configPath)
    : gradientTexture(gradientTexture)
    , configPath(configPath)
{

}

bool GDebugRendererPPG::LoadSDTree(SDTree& sdTree, 
                                   const nlohmann::json& config,
                                   const std::string& configPath,
                                   uint32_t depth)
{
    auto loc = config.find(SD_TREE_NAME);
    if(loc == config.end()) return false;
    if(loc->size() >= depth) return false;

    std::string fileName = (*loc)[depth];
    std::string fileMergedPath = Utility::MergeFileFolder(configPath, fileName);
    std::ifstream file(fileMergedPath, std::ios::binary);
    
    // Read STree Start Offset
    uint64_t sTreeOffset;
    // ...
    // Read STree Node Count
    uint64_t sTreeNodeCount;
    // ...
    // Read DTree Count
    uint64_t dTreeCount;
    // ...
    // Read DTree Offset/Count Pairs
    std::vector<Vector2ul> offsetCountPairs;
    // ...
    // Read STree
    sdTree.extents;
    // ...
    // Read DTrees in order
    for(uint64_t i = 0; i < dTreeCount; i++)
    {
        // Read Base
        std::pair<uint32_t, float> dTreeBase;
        // ...
        // Read Nodes
        std::vector<DTreeNode> dTreeNodes;
        // ...

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

void GDebugRendererPPG::RenderDirectional(TextureGL&, const Vector3f& worldPos, uint32_t depth)
{
    // Find DTree
    uint32_t dTreeIndex = sdTrees[depth].FindDTree(worldPos);

    // Buffer the Nodes for GPU
    //




    // Render Filled Squares

    // Then Render lines for visual clarity
    // TODO: .....
}

