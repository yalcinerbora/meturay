#include "SimpleTracerSetup.h"

TEST(SimpleTracerTests, HelloTriangle)
{
    EnableVTMode();
    SimpleTracerSetup setup(u8"TestScenes/helloTriangle.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracerTests, HelloSphere)
{
    EnableVTMode();
    SimpleTracerSetup setup(u8"TestScenes/helloSphere.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracerTests, HelloBox)
{
    EnableVTMode();
    SimpleTracerSetup setup(u8"TestScenes/helloBox.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

//TEST(SimpleTracerTests, OcclusionTest)
//{
//    EnableVTMode();
//    SimpleTracerSetup setup(u8"TestScenes/occlusionTest.json", 0.0);
//    ASSERT_TRUE(setup.Init());
//    setup.Body();
//}
//
//TEST(SimpleTracerTests, CullTest)
//{
//    EnableVTMode();
//    SimpleTracerSetup setup(u8"TestScenes/cullTest.json", 0.0);
//    ASSERT_TRUE(setup.Init());
//    setup.Body();
//}

TEST(SimpleTracerTests, CornellBox)
{
    EnableVTMode();
    SimpleTracerSetup setup(u8"TestScenes/helloCornell.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracerTests, Bunny)
{
    EnableVTMode();
    SimpleTracerSetup setup(u8"TestScenes/helloBunny.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracerTests, Dragon)
{
    EnableVTMode();
    SimpleTracerSetup setup(u8"TestScenes/helloDragon.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}