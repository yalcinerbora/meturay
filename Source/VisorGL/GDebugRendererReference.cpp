#include "GDebugRendererReference.h"

#include "RayLib/SceneIO.h"
#include "RayLib/FileSystemUtility.h"
#include "RayLib/StringUtility.h"

#include <regex>

void GDebugRendererRef::LoadPaths(const Vector2i& resolution,
                                  const std::string& pathRegex)
{
    static constexpr std::string_view RES_TOKEN = "[%]";
    static constexpr std::string_view REGEX = "\\[[0-9]+, [0-9]+\\]";

    // Generate Regex for the image names
    std::string regexStr = pathRegex;
    size_t locPath = regexStr.find_first_of(RES_TOKEN);
    regexStr.replace(locPath, RES_TOKEN.length(), REGEX);
    // Generate Regex for file name only
    std::string fileRegexStr = Utility::PathFile(pathRegex);
    size_t locFile = fileRegexStr.find_first_of(RES_TOKEN);
    fileRegexStr.replace(locFile, RES_TOKEN.length(), REGEX);

    // Dont forget to regexify extension '.'
    Utility::ReplaceAll(regexStr, ".", "\\.");
    Utility::ReplaceAll(fileRegexStr, ".", "\\.");


    // Generate Actual regex 
    std::regex regexFull(regexStr);
    std::regex regexFileOnly(fileRegexStr);
    // List all files that match to the regex
    std::vector<std::string> files = Utility::ListFilesInFolder(Utility::PathFolder(pathRegex), 
                                                                regexFileOnly);

    // Now parse and sort these files
    auto ParsePixelId = [] (const std::string& path)
    {
        std::string fileName = Utility::PathFile(path);

        // No need to have a while loop it should have only one match
        std::smatch match;
        std::regex_search(fileName, match, std::regex(REGEX.data()));
        std::string pixelPortion = match.str();

        // Dont forget that pixels are ordered reverse
        // to be sorted properly
        Vector2i res;
        res[1] = std::stoi(pixelPortion.substr(pixelPortion.find_first_of('[') + 1,
                                               pixelPortion.find_first_of(',')));
        res[0] = std::stoi(pixelPortion.substr(pixelPortion.find_first_of(", ") + 1,
                                               pixelPortion.find_first_of(']')));

        

        return res;
    };

    // Sort the files using a map
    std::map<int32_t, std::string> orderedPaths;
    for(const std::string& file : files)
    {
        Vector2i pixel = ParsePixelId(file);
        // Skip out of range data
        if((pixel[0] < resolution[0]) &&
           (pixel[1] < resolution[1]))
        {
            int key = pixel[1] * resolution[0] + pixel[0];
            orderedPaths.emplace(key, file);
        }
    }
    // Finally Push sorted data to a vector
    referencePaths.reserve(orderedPaths.size());
    for(const auto& i : orderedPaths)
    {
        referencePaths.push_back(i.second);
    }

    assert((resolution[0] * resolution[1]) == referencePaths.size());
}

GDebugRendererRef::GDebugRendererRef(const nlohmann::json& config,
                                     const TextureGL& gradTex)
    : gradientTex(gradTex)
{
    resolution = SceneIO::LoadVector<2, int32_t>(config[RESOLUTION_NAME]);
    std::string pathRegex = SceneIO::LoadString(config[IMAGES_NAME]);
    // Generate Image Paths
    LoadPaths(resolution, pathRegex);
}

void GDebugRendererRef::RenderSpatial(TextureGL& tex) const
{
    // TODO: implement
}

void GDebugRendererRef::RenderDirectional(TextureGL& tex,
                                          const Vector2i& pixel,
                                          const Vector2i& refResolution) const
{
    // Convert pixel Location to the local pixel
    Vector2f ratio = (Vector2f(resolution[0], resolution[1]) /
                          Vector2f(refResolution[0], refResolution[1]));

    Vector2f mappedPix = Vector2f(pixel[0], pixel[1]) * ratio;

    Vector2i pixelInt = Vector2i(mappedPix[0], mappedPix[1]);
    uint32_t pixelLinear = resolution[0] * pixelInt[1] + pixelInt[0];

    const std::string& file = referencePaths[pixelLinear];
    //TextureGL exrTex = TextureGL(file);
    tex = std::move(TextureGL(file));

    //TODO: 
}