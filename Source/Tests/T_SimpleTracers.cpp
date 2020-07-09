#include "SimpleTracerSetup.h"

TEST(SimpleTracers, HelloTriangle)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestDirect", u8"TestScenes/helloTriangle.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, HelloSphere)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestDirect", u8"TestScenes/helloSphere.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, HelloBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestDirect", u8"TestScenes/helloBox.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, BVHTest)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestDirect", u8"TestScenes/bvhTest.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracers, BaseBVHTest)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestDirect", u8"TestScenes/baseBVHTest.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}