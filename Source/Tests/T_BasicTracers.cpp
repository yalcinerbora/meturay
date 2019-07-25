#include "SimpleTracerSetup.h"

TEST(SimpleTracerTests, HelloTriangle)
{
    EnableVTMode();

    SimpleTracerSetup setup("TestScenes/helloTriangle.json",
                            0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracerTests, HelloSphere)
{
    EnableVTMode();

    SimpleTracerSetup setup("TestScenes/helloSphere.json",
                            0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(SimpleTracerTests, HelloBox)
{
    EnableVTMode();

    SimpleTracerSetup setup("TestScenes/helloBox.json",
                            0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

//TEST(SimpleTracerTests, CornellBox)
//{
//    EnableVTMode();
//
//    SimpleTracerSetup setup("TestScenes/cornellBox.json",
//                            0.0);
//    setup.Init();
//    setup.Body();
//}