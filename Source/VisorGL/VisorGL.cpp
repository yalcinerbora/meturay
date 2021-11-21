#include "ImageIO/EntryPoint.h"

#include "VisorGL.h"

#include "RayLib/Log.h"
#include "RayLib/CPUTimer.h"
#include "RayLib/VisorError.h"
#include "RayLib/VisorInputI.h"

#include "GLConversionFunctions.h"
#include "GLFWCallbackDelegator.h"

#include <map>
#include <cassert>
#include <thread>

void VisorGL::OGLCallbackRender(GLenum,
                                GLenum type,
                                GLuint id,
                                GLenum severity,
                                GLsizei,
                                const char* message,
                                const void*)
{
    GLFWCallbackDelegator::OGLDebugLog(type, id, severity, message);
}

void VisorGL::ReallocImages()
{
    // Textures
    // Buffered output textures
    glDeleteTextures(2, outputTextures);
    glGenTextures(2, outputTextures);
    for(int i = 0; i < 2; i++)
    {
        glBindTexture(GL_TEXTURE_2D, outputTextures[i]);
        glTexStorage2D(GL_TEXTURE_2D, 1, PixelFormatToSizedGL(imagePixFormat),
                       imageSize[0], imageSize[1]);
    }
    // Sample count texture
    glDeleteTextures(1, &sampleCountTexture);
    glGenTextures(1, &sampleCountTexture);
    glBindTexture(GL_TEXTURE_2D, sampleCountTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI,
                   imageSize[0], imageSize[1]);
    // Buffer input texture
    glDeleteTextures(1, &bufferTexture);
    glGenTextures(1, &bufferTexture);
    glBindTexture(GL_TEXTURE_2D, bufferTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, PixelFormatToSizedGL(imagePixFormat),
                   imageSize[0], imageSize[1]);
    // Sample input texture
    glDeleteTextures(1, &sampleTexture);
    glGenTextures(1, &sampleTexture);
    glBindTexture(GL_TEXTURE_2D, sampleTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI,
                   imageSize[0], imageSize[1]);
    // SDR Texture
    glDeleteTextures(1, &sdrTexture);
    glGenTextures(1, &sdrTexture);
    glBindTexture(GL_TEXTURE_2D, sdrTexture);
    glTexStorage2D(GL_TEXTURE_2D, 1, PixelFormatToSizedGL(PixelFormat::RGBA8_UNORM),
                   imageSize[0], imageSize[1]);
}

void VisorGL::ProcessCommand(const VisorGLCommand& c)
{
    const Vector2i inSize = c.end - c.start;

    switch(c.type)
    {
        case VisorGLCommand::REALLOC_IMAGES:
        {
            // Do realloc images
            ReallocImages();
            // Let the case fall to reset image
            // since we just allocated and need reset on image
            // as well.
            [[fallthrough]];
        }
        case VisorGLCommand::RESET_IMAGE:
        {
            // Just clear the sample count to zero
            const GLuint clearDataInt = 0;
            const Vector4f clearDataFloat = Vector4f(0.0f);
            glClearTexSubImage(sampleCountTexture, 0,
                               c.start[0], c.start[1], 0,
                               inSize[0], inSize[1], 1,
                               GL_RED_INTEGER, GL_UNSIGNED_INT, &clearDataInt);
            break;
        }
        case VisorGLCommand::SET_PORTION:
        {
            // Copy (Let OGL do the conversion)
            glBindTexture(GL_TEXTURE_2D, sampleTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0,
                            0, 0,
                            inSize[0], inSize[1],
                            GL_RED_INTEGER,
                            GL_UNSIGNED_INT,
                            c.data.data() + c.offset);
            glBindTexture(GL_TEXTURE_2D, bufferTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0,
                            0, 0,
                            inSize[0], inSize[1],
                            PixelFormatToGL(c.format),
                            PixelFormatToTypeGL(c.format),
                            c.data.data());

            // After Copy
            // Accumulate data
            int nextIndex = (currentIndex + 1) % 2;
            GLuint outTexture = outputTextures[nextIndex];
            GLuint inTexture = outputTextures[currentIndex];
            // Shader
            compAccum.Bind();
            // Textures
            glActiveTexture(GL_TEXTURE0 + T_IN_BUFFER);
            glBindTexture(GL_TEXTURE_2D, bufferTexture);
            glActiveTexture(GL_TEXTURE0 + T_IN_SAMPLE);
            glBindTexture(GL_TEXTURE_2D, sampleTexture);
            glActiveTexture(GL_TEXTURE0 + T_IN_COLOR);
            glBindTexture(GL_TEXTURE_2D, inTexture);

            // Images
            glBindImageTexture(I_SAMPLE, sampleCountTexture,
                               0, false, 0, GL_READ_WRITE, GL_R32I);
            glBindImageTexture(I_OUT_COLOR, outTexture,
                               0, false, 0, GL_WRITE_ONLY,
                               PixelFormatToSizedGL(imagePixFormat));

            // Uniforms
            glUniform2iv(U_RES, 1, static_cast<const int*>(imageSize));
            glUniform2iv(U_START, 1, static_cast<const int*>(c.start));
            glUniform2iv(U_END, 1, static_cast<const int*>(c.end));

            // Call for entire image (we also copy image)
            //
            GLuint gridX = (imageSize[0] + 16 - 1) / 16;
            GLuint gridY = (imageSize[1] + 16 - 1) / 16;
            glDispatchCompute(gridX, gridY, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                            GL_TEXTURE_FETCH_BARRIER_BIT);

            // Swap input and output
            currentIndex = nextIndex;
            break;
        }
        case VisorGLCommand::SAVE_IMAGE:
        {
            // TODO: load as 8-bit color (but alignment trolls i think heap gets corrupted)
            std::vector<Vector3uc> pixels(imageSize[0] * imageSize[1]);
            GLuint readTexture = sdrTexture;
            // Thightly pack pixels for reading
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glBindTexture(GL_TEXTURE_2D, readTexture);
            glGetnTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE,
                           static_cast<GLsizei>(pixels.size() * sizeof(Vector4uc)),
                           pixels.data());

            ImageIOError e = ImageIOError::OK;
            const ImageIOI& io = *ImageIOInstance();
            if((e = io.WriteImage(pixels,
                                  Vector2ui(imageSize[0], imageSize[1]),
                                  PixelFormat::RGB8_UNORM, ImageType::PNG,
                                  "imgOut.png")) != ImageIOError::OK)
            {
                METU_ERROR_LOG(static_cast<std::string>(e));
            }
            break;
        }
        case VisorGLCommand::SAVE_IMAGE_HDR:
        {
            std::vector<Vector3f> pixels(imageSize[0] * imageSize[1]);
            GLuint readTexture = outputTextures[currentIndex];
            glBindTexture(GL_TEXTURE_2D, readTexture);
            // Thightly pack pixels for reading
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT,
                          pixels.data());

            ImageIOError e = ImageIOError::OK;
            const ImageIOI& io = *ImageIOInstance();
            if((e = io.WriteImage(pixels,
                                  Vector2ui(imageSize[0], imageSize[1]),
                                  PixelFormat::RGB_FLOAT, ImageType::EXR,
                                  "imgOut.exr")) != ImageIOError::OK)
            {
                METU_ERROR_LOG(static_cast<std::string>(e));
            }
            break;
        }
    };
}

void VisorGL::RenderImage()
{
    // Clear
    Vector2i size;
    if(viewportSize.CheckChanged(size))
    {
        glViewport(0, 0, size[0], size[1]);
    }

    // Clear Buffer
    glClearColor(0.0f, 0.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Bind Shaders
    vertPP.Bind();
    fragPP.Bind();

    // Bind Texture
    glActiveTexture(GL_TEXTURE0 + T_IN_COLOR);
    glBindTexture(GL_TEXTURE_2D, sdrTexture);

    // Draw
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

void VisorGL::SetFBSizeFromInput(const Vector2i& fbSize)
{
    viewportSize = fbSize;
}

void VisorGL::SetWindowSizeFromInput(const Vector2i& winSize)
{
    vOpts.wSize = winSize;
}

void VisorGL::SetOpenStateFromInput(bool b)
{
    open = b;
}

VisorInputI* VisorGL::InputInterface()
{
    return input;
}

VisorGL::VisorGL(const VisorOptions& opts,
                 const Vector2i& imgRes,
                 const PixelFormat& imagePixelFormat)
    : input(nullptr)
    , window(nullptr)
    , open(false)
    , commandList(opts.eventBufferSize)
    , outputTextures{0, 0}
    , sampleCountTexture(0)
    , bufferTexture(0)
    , sampleTexture(0)
    , sdrTexture(0)
    , currentIndex(0)
    , linearSampler(0)
    , nearestSampler(0)
    , vBuffer(0)
    , vao(0)
    , vOpts(opts)
    , imageSize(imgRes)
    , imagePixFormat(imagePixelFormat)
    , visorGUI(nullptr)
{}

VisorGL::~VisorGL()
{
    // Tonemapper
    toneMapGL = ToneMapGL();

    // Delete Vertex Arrays & Buffers
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vBuffer);

    // Delete Shaders
    vertPP = ShaderGL();
    fragPP = ShaderGL();
    compAccum = ShaderGL();

    // Delete Samplers
    glDeleteSamplers(1, &linearSampler);
    glDeleteSamplers(1, &nearestSampler);

    // Delete Textures
    glDeleteTextures(1, &bufferTexture);
    glDeleteTextures(1, &sampleTexture);
    glDeleteTextures(1, &sampleCountTexture);
    glDeleteTextures(2, outputTextures);
    glDeleteTextures(2, &sdrTexture);

    if(visorGUI) visorGUI = nullptr;

    if(window != nullptr) glfwDestroyWindow(window);
    GLFWCallbackDelegator::Instance().DetachWindow(window);
}

bool VisorGL::IsOpen()
{
    return open;
}

void VisorGL::ProcessInputs()
{
    glfwPollEvents();
}

void VisorGL::Render()
{
    Utility::CPUTimer t;
    if(vOpts.fpsLimit > 0.0f) t.Start();

    // Consume commands
    // TODO: optimize this skip multiple reset commands
    // just process the last and other commands afterwards
    bool imageModified = false;
    VisorGLCommand command;
    while(commandList.TryDequeue(command))
    {
        // Image is considered modified if at least one command is
        // processed on this frame
        imageModified = true;
        ProcessCommand(command);
    }

    // Do Tone Map
    // Only do tone map if HDR image is modified
    if(imageModified)
    {
        ToneMapOptions tmOpts;
        if(!vOpts.enableTMO)
        {
            tmOpts.doKeyAdjust = false;
            tmOpts.doGamma = false;
            tmOpts.doToneMap = false;
        }
        else if(visorGUI)
            tmOpts = visorGUI->TMOptions();
        else
            tmOpts = DefaultTMOptions;

        //tmOpts.key = 0.01f;
        tmOpts.doKeyAdjust = false;

        // Always call this even if there are no parameters
        // set to do tone mapping since this function
        // will write to sdr image and RenderImage function
        // will use it to present it to the FB
        toneMapGL.ToneMap(sdrTexture,
                          PixelFormat::RGBA8_UNORM,
                          outputTextures[currentIndex],
                          tmOpts, imageSize);
    }

    // Render Image
    RenderImage();

    // After Render GUI
    if(vOpts.enableGUI) visorGUI->Render(sdrTexture,
                                         imageSize);

    // Finally Swap Buffers
    glfwSwapBuffers(window);

    // TODO: This is kinda wrong?? check it
    // since it does not excatly makes it to a certain FPS value
    if(vOpts.fpsLimit > 0.0f)
    {
        t.Stop();
        double sleepMS = (1000.0 / vOpts.fpsLimit);
        sleepMS -= t.Elapsed<std::milli>();
        sleepMS = std::max(0.0, sleepMS);
        if(sleepMS != 0.0)
        {
            std::chrono::duration<double, std::milli> chronoMillis(sleepMS);
            std::this_thread::sleep_for(chronoMillis);
        }
    }
}

void VisorGL::SetImageRes(Vector2i resolution)
{
    imageSize = resolution;

    VisorGLCommand command = {};
    command.type = VisorGLCommand::REALLOC_IMAGES;
    command.start = Zero2i;
    command.end = imageSize;
    commandList.Enqueue(std::move(command));
}

void VisorGL::SetImageFormat(PixelFormat format)
{
    imagePixFormat = format;

    VisorGLCommand command = {};
    command.type = VisorGLCommand::REALLOC_IMAGES;
    command.start = Zero2i;
    command.end = imageSize;
    commandList.Enqueue(std::move(command));
}

void VisorGL::ResetSamples(Vector2i start, Vector2i end)
{
    end = Vector2i::Min(end, imageSize);

    VisorGLCommand command;
    command.type = VisorGLCommand::RESET_IMAGE;
    command.format = imagePixFormat;
    command.start = start;
    command.end = end;

    commandList.Enqueue(std::move(command));
}

void VisorGL::AccumulatePortion(const std::vector<Byte> data,
                                PixelFormat f, size_t offset,
                                Vector2i start,
                                Vector2i end)
{
    end = Vector2i::Min(end, imageSize);

    VisorGLCommand command;
    command.type = VisorGLCommand::SET_PORTION;
    command.start = start;
    command.end = end;
    command.format = f;
    command.offset = offset;
    command.data = std::move(data);

    commandList.Enqueue(std::move(command));
}

const VisorOptions& VisorGL::VisorOpts() const
{
    return vOpts;
}

void VisorGL::SetWindowSize(const Vector2i& size)
{
    glfwSetWindowSize(window, size[0], size[1]);
}

void VisorGL::SetFPSLimit(float f)
{
    vOpts.fpsLimit = f;
}

Vector2i VisorGL::MonitorResolution() const
{
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(primaryMonitor);

    return Vector2i(mode->width, mode->height);
}

void VisorGL::SetTransform(const VisorTransform& c)
{
    if(input) input->SetTransform(c);
}

void VisorGL::SetSceneCameraCount(uint32_t c)
{
    if(input) input->SetSceneCameraCount(c);
}

void VisorGL::SetRenderingContextCurrent()
{
    glfwMakeContextCurrent(window);

    // TODO: temp fix for multi threaded visor
    // GLEW functions does not? accesible between threads?
    // It works on Maxwell but it did not work on
    // Pascal (GTX 1050 is pascal?).

    // Also this is not a permanent solution
    // Program should not reload entire gl functions
    // on every context set
    // Threaded visor programs should set this once anyway
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if(err != GLEW_OK)
    {
        METU_ERROR_LOG("{:s}", glewGetErrorString(err));
    }
}

void VisorGL::ReleaseRenderingContext()
{
    glfwMakeContextCurrent(nullptr);
}

VisorError VisorGL::Initialize(VisorInputI& vInput)
{
    input = &vInput;

    GLFWCallbackDelegator& glfwCallback = GLFWCallbackDelegator::Instance();

    // Common Window Hints
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

    // This was buggy on nvidia cards couple of years ago
    // So instead manually convert image using
    // computer shader or w/e sRGB space
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_FALSE);

    glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_FLUSH);

    if(vOpts.stereoOn)
        glfwWindowHint(GLFW_STEREO, GLFW_TRUE);

    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, IS_DEBUG_MODE);

    // At most 16x MSAA
    //glfwWindowHint(GLFW_SAMPLES, 16);

    // Pixel of WindowFBO
    // Full precision output
    // TODO: make it to utilize vOpts.wFormat
    int rBits = 0,
        gBits = 0,
        bBits = 0,
        aBits = 0;
    switch(vOpts.wFormat)
    {
        case PixelFormat::RGBA8_UNORM:
            aBits = 8;
        case PixelFormat::RGB8_UNORM:
            bBits = 8;
        case PixelFormat::RG8_UNORM:
            gBits = 8;
        case PixelFormat::R8_UNORM:
            rBits = 8;
            break;
        case PixelFormat::RGBA16_UNORM:
            aBits = 16;
        case PixelFormat::RGB16_UNORM:
            bBits = 16;
        case PixelFormat::RG16_UNORM:
            gBits = 16;
        case PixelFormat::R16_UNORM:
            rBits = 16;
            break;
        case PixelFormat::RGBA_HALF:
            aBits = 16;
        case PixelFormat::RGB_HALF:
            bBits = 16;
        case PixelFormat::RG_HALF:
            gBits = 16;
        case PixelFormat::R_HALF:
            rBits = 16;
            break;
        case PixelFormat::RGBA_FLOAT:
            aBits = 32;
        case PixelFormat::RGB_FLOAT:
            bBits = 32;
        case PixelFormat::RG_FLOAT:
            gBits = 32;
        case PixelFormat::R_FLOAT:
            rBits = 32;
            break;
        default:
            aBits = 8;
            bBits = 8;
            gBits = 8;
            rBits = 8;
            break;
    }
    glfwWindowHint(GLFW_RED_BITS, rBits);
    glfwWindowHint(GLFW_GREEN_BITS, gBits);
    glfwWindowHint(GLFW_BLUE_BITS, bBits);
    glfwWindowHint(GLFW_ALPHA_BITS, aBits);

    // No depth buffer or stencil buffer etc
    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);

    window = glfwCreateWindow(vOpts.wSize[0],
                              vOpts.wSize[1],
                              "METUray Visor",
                              nullptr,
                              nullptr);
    if(window == nullptr)
    {
        return VisorError::WINDOW_GENERATION_ERROR;
    }


    // Set Callbacks
    vInput.SetVisor(*this);
    glfwCallback.AttachWindow(window, this);

    glfwMakeContextCurrent(window);
    // Initial Option set
    glfwSwapInterval((vOpts.vSyncOn) ? 1 : 0);

    // Now Init GLEW
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if(err != GLEW_OK)
    {
        METU_ERROR_LOG("{:s}", glewGetErrorString(err));
        return VisorError::RENDER_FUCTION_GENERATOR_ERROR;
    }

    // Print Stuff Now
    // Window Done
    METU_LOG("Window Initialized.");
    METU_LOG("GLEW\t: {:s}", glewGetString(GLEW_VERSION));
    METU_LOG("GLFW\t: {:s}", glfwGetVersionString());
    METU_LOG("");
    METU_LOG("Renderer Information...");
    METU_LOG("OpenGL\t: {:s}", glGetString(GL_VERSION));
    METU_LOG("GLSL\t: {:s}", glGetString(GL_SHADING_LANGUAGE_VERSION));
    METU_LOG("Device\t: {:s}", glGetString(GL_RENDERER));
    METU_LOG("");

    if constexpr(IS_DEBUG_MODE)
    {
        // Add Callback
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(VisorGL::OGLCallbackRender, nullptr);
        glDebugMessageControl(GL_DONT_CARE,
                              GL_DONT_CARE,
                              GL_DONT_CARE,
                              0,
                              nullptr,
                              GL_TRUE);
    }

    // Shaders
    vertPP = ShaderGL(ShaderType::VERTEX, u8"Shaders/PProcessGeneric.vert");
    fragPP = ShaderGL(ShaderType::FRAGMENT, u8"Shaders/PProcessGeneric.frag");
    compAccum = ShaderGL(ShaderType::COMPUTE, u8"Shaders/AccumInput.comp");

    ReallocImages();

    // ToneMaper
    toneMapGL = ToneMapGL(true);

    // Sampler
    glGenSamplers(1, &linearSampler);
    glSamplerParameteri(linearSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glSamplerParameteri(linearSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glSamplerParameteri(linearSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(linearSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(linearSampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glGenSamplers(1, &nearestSampler);
    glSamplerParameteri(nearestSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(nearestSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(nearestSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(nearestSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(nearestSampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    // Buffer
    glGenBuffers(1, &vBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, vBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6, PostProcessTriData, GL_STATIC_DRAW);

    // Vertex Buffer
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindVertexBuffer(0, vBuffer, 0, sizeof(float) * 2);
    glEnableVertexAttribArray(IN_POS);
    glVertexAttribFormat(IN_POS, 2, GL_FLOAT, false, 0);
    glVertexAttribBinding(IN_POS, IN_POS);

    // Pre-Bind Everything
    // States
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    // Intel OGL complaints about this as redundant call?
    //glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // FBO
    glColorMask(true, true, true, true);
    glDepthMask(false);
    glStencilMask(false);

    // Sampler
    glBindSampler(T_IN_COLOR, linearSampler);
    glBindSampler(T_IN_BUFFER, linearSampler);
    glBindSampler(T_IN_SAMPLE, nearestSampler);

    // Bind VAO
    glBindVertexArray(vao);

    // Finally Show Window
    glfwShowWindow(window);
    open = true;

    // CheckGUI and Initialize
    if(vOpts.enableGUI)
        visorGUI = std::make_unique<VisorGUI>(window);

    // Unmake context current on this
    // thread after initialization
    glfwMakeContextCurrent(nullptr);
    return VisorError::OK;
}

void VisorGL::SaveImage(bool saveAsHDR)
{

    VisorGLCommand command;
    command.type = saveAsHDR
                    ? VisorGLCommand::SAVE_IMAGE_HDR
                    : VisorGLCommand::SAVE_IMAGE;

    commandList.Enqueue(std::move(command));
}