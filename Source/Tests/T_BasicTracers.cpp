#include "SimpleTracerSetup.h"

TEST(SimpleTracerTests, HelloTriangle)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestDirect", u8"TestScenes/helloTriangle.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracerTests, HelloSphere)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestDirect", u8"TestScenes/helloSphere.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracerTests, HelloBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestDirect", u8"TestScenes/helloBox.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracerTests, BVHTest)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestDirect", u8"TestScenes/bvhTest.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracerTests, BaseBVHTest)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestDirect", u8"TestScenes/baseBVHTest.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}