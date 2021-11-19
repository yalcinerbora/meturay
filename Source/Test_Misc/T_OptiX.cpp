#include "SimpleTracerSetup.h"

TEST(OptiX, Empty)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", u8"TestScenes/emptySceneLinear.json", 0.0,
                            false, true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(OptiX, HelloTriangle)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", u8"TestScenes/helloTriangle.json", 0.0,
                            false, true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(OptiX, HelloSphere)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", u8"TestScenes/helloSphere.json", 0.0,
                            false, true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(OptiX, HelloBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", u8"TestScenes/helloBox.json", 0.0,
                            false, true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(OptiX, Accelerator)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", u8"TestScenes/bvhTest.json", 0.0,
                            false, true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(OptiX, BaseAccelerator)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", u8"TestScenes/baseBVHTest.json", 0.0,
                            false, true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(OptiX, AnyHit)
{
    EnableVTMode();
    SimpleTracerSetup setup("DirectTracer", u8"TestScenes/alphaMapTest.json", 0.0,
                            false, true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}