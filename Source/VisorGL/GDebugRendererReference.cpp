#include "GDebugRendererReference.h"
#include "GLConversionFunctions.h"

#include "RayLib/SceneIO.h"
#include "RayLib/FileSystemUtility.h"
#include "RayLib/StringUtility.h"
#include "RayLib/Log.h"

#include "ImageIO/EntryPoint.h"

#include <Imgui/imgui.h>
#include "GuideDebugGUIFuncs.h"

#include <regex>

void GDebugRendererRef::LoadPaths(const Vector2i& res,
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

    // Don't forget to regexify extension '.'
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

        // Don't forget that pixels are ordered reverse
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
        if((pixel[0] < res[0]) &&
           (pixel[1] < res[1]))
        {
            int key = pixel[1] * res[0] + pixel[0];
            orderedPaths.emplace(key, file);
        }
    }
    // Finally Push sorted data to a vector
    referencePaths.reserve(orderedPaths.size());
    for(const auto& i : orderedPaths)
    {
        referencePaths.push_back(i.second);
    }

    assert(static_cast<uint32_t>(res[0] * res[1]) == referencePaths.size());
}

GDebugRendererRef::GDebugRendererRef(const nlohmann::json& config,
                                     const std::string& configPath,
                                     const TextureGL& gradTex)
    : compReduction(ShaderType::COMPUTE, u8"Shaders/TextureMaxReduction.comp")
    , compRefRender(ShaderType::COMPUTE, u8"Shaders/PGReferenceRender.comp")
    , linearSampler(SamplerGLEdgeResolveType::CLAMP,
                    SamplerGLInterpType::LINEAR)
    , gradientTex(gradTex)
{
    resolution = SceneIO::LoadVector<2, int32_t>(config[RESOLUTION_NAME]);
    std::string pathRegex = config[IMAGES_NAME];
    pathRegex = Utility::MergeFileFolder(configPath, pathRegex);
    // Generate Image Paths
    LoadPaths(resolution, pathRegex);
}

void GDebugRendererRef::RenderSpatial(TextureGL&) const
{
    // TODO: implement
}

void GDebugRendererRef::UpdateDirectional(bool doLogScale,
                                          const Vector2i& worldPixel,
                                          const Vector2i& worldRefResolution)
{
    // Convert pixel Location to the local pixel
    Vector2f ratio = (Vector2f(resolution[0], resolution[1]) /
                      Vector2f(worldRefResolution[0], worldRefResolution[1]));
    Vector2f mappedPix = Vector2f(worldPixel[0], worldPixel[1]) * ratio;
    Vector2i pixelInt = Vector2i(mappedPix[0], mappedPix[1]);
    uint32_t pixelLinear = resolution[0] * pixelInt[1] + pixelInt[0];

    const std::string& file = referencePaths[pixelLinear];

    // Temp Load Lum Texture
    Vector2ui dim;
    PixelFormat pf;
    std::vector<Byte> pixels;
    ImageIOInstance()->ReadImage(pixels, pf, dim, file);

    TextureGL lumTexture = TextureGL(dim, pf);
    lumTexture.CopyToImage(pixels, Zero2ui, dim, pf);
    if(lumTexture.Size() != currentTexture.Size())
    {
        currentTexture = TextureGL(lumTexture.Size(), PixelFormat::RGBA8_UNORM);
    }

    // Copy data to actual vector
    assert(pf == PixelFormat::R_FLOAT ||
           pf == PixelFormat::R_HALF);
    size_t linearSize = dim[0] * dim[1];
    currentValues.resize(linearSize);
    ImageIOError e = ImageIOInstance()->ConvertPixels(reinterpret_cast<Byte*>(currentValues.data()),
                                                      PixelFormat::R_FLOAT,
                                                      pixels.data(), pf,
                                                      dim);
    if(e != ImageIOError::OK)
    {
        METU_ERROR_LOG(static_cast<std::string>(e));
    };

    // Get a max luminance buffer;
    float initalMaxData = 0.0f;
    GLuint maxBuffer;
    glGenBuffers(1, &maxBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, maxBuffer);
    glBufferStorage(GL_SHADER_STORAGE_BUFFER, 1 * sizeof(float), &initalMaxData, 0);

    // Both of these compute shaders total work count is same
    const GLuint workCount = lumTexture.Size()[1] * lumTexture.Size()[0];
    // Some WG Definitions (statically defined in shader)
    static constexpr GLuint WORK_GROUP_1D_X = 256;
    static constexpr GLuint WORK_GROUP_2D_X = 16;
    static constexpr GLuint WORK_GROUP_2D_Y = 16;

    // =======================================================
    // Set Max Shader
    compReduction.Bind();
    // Bind Uniforms
    glUniform2ui(U_RES, lumTexture.Size()[0], lumTexture.Size()[1]);
    // Bind SSBO
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSB_MAX_LUM, maxBuffer);
    // Textures
    lumTexture.Bind(T_IN_LUM_TEX);
    // Dispatch Max Shader
    // Max shader is 1D shader set data accordingly
    GLuint gridX_1D = (workCount + WORK_GROUP_1D_X - 1) / WORK_GROUP_1D_X;
    glDispatchCompute(gridX_1D, 1, 1);
    glMemoryBarrier(GL_UNIFORM_BARRIER_BIT |
                    GL_SHADER_STORAGE_BARRIER_BIT);
    // =======================================================
    // Unbind SSBO just to be sure
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSB_MAX_LUM, 0);

    //// Debug check of the reduced value
    //float v;
    //glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    //glBindBuffer(GL_COPY_READ_BUFFER, maxBuffer);
    //glGetBufferSubData(GL_COPY_READ_BUFFER, 0, sizeof(float), &v);
    //METU_LOG("Max {:f}", v);

    // =======================================================
    // Set Render Shader
    compRefRender.Bind();
    // Bind Uniforms
    glUniform2ui(U_RES, lumTexture.Size()[0], lumTexture.Size()[1]);
    glUniform1i(U_LOG_ON, doLogScale ? 1 : 0);
    //
    // UBOs
    glBindBufferBase(GL_UNIFORM_BUFFER, UB_MAX_LUM, maxBuffer);
    // Textures
    lumTexture.Bind(T_IN_LUM_TEX);
    gradientTex.Bind(T_IN_GRAD_TEX);  linearSampler.Bind(T_IN_GRAD_TEX);
    // Images
    glBindImageTexture(I_OUT_REF_IMAGE, currentTexture.TexId(),
                       0, false, 0, GL_WRITE_ONLY,
                       PixelFormatToSizedGL(currentTexture.Format()));
    // Dispatch Render Shader
    // Max shader is 2D shader set data accordingly
    GLuint gridX_2D = (lumTexture.Size()[0] + WORK_GROUP_2D_X - 1) / WORK_GROUP_2D_X;
    GLuint gridY_2D = (lumTexture.Size()[1] + WORK_GROUP_2D_Y - 1) / WORK_GROUP_2D_Y;
    glDispatchCompute(gridX_2D, gridY_2D, 1);
    // =======================================================
    // All done!!!

    // Delete Temp Max Buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glDeleteBuffers(1, &maxBuffer);
}

void GDebugRendererRef::RenderGUI(const ImVec2& windowSize)
{
    ImGui::BeginChild("refPGTexture", windowSize, false);
    ImGui::SameLine(0.0f, GuideDebugGUIFuncs::CenteredTextLocation(REFERENCE_TEXT, windowSize.x));
    ImGui::Text(REFERENCE_TEXT);
    ImVec2 remainingSize = GuideDebugGUIFuncs::FindRemainingSize(windowSize);
    remainingSize.x = remainingSize.y;
    ImGui::NewLine();
    ImGui::SameLine(0.0f, (windowSize.x - remainingSize.x) * 0.5f - ImGui::GetStyle().WindowPadding.x);
    // Debug Reference Image
    if(currentTexture.Size() != Zero2ui)
        GuideDebugGUIFuncs::RenderImageWithZoomTooltip(currentTexture, currentValues, remainingSize);
    else
        ImGui::Dummy(remainingSize);
    ImGui::EndChild();
}