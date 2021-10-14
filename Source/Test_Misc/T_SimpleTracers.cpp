#include "SimpleTracerSetup.h"

TEST(SimpleTracers, EmptyLinear)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", false,
                            u8"TestScenes/emptySceneLinear.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, EmptyBVH)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", false,
                            u8"TestScenes/emptySceneBVH.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, HelloTriangle)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", false,
                            u8"TestScenes/helloTriangle.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, HelloSphere)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", false,
                            u8"TestScenes/helloSphere.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, HelloBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", false,
                            u8"TestScenes/helloBox.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, BVHTest)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", false,
                            u8"TestScenes/bvhTest.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, BaseBVHTest)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", false,
                            u8"TestScenes/baseBVHTest.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, QuatInterpTest)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", false,
                            u8"TestScenes/quatInterp.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, TextureTest)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", false,
                            u8"TestScenes/textureTest.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, SkySphere)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", false,
                            u8"TestScenes/skySphereHDR.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, AlphaMap)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", false,
                            u8"TestScenes/alphaMapTest.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}