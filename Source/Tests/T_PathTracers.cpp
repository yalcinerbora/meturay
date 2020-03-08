#include "SimpleTracerSetup.h"


TEST(PathTracerTests, CornellBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestPath", u8"TestScenes/helloCornell.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracerTests, Bunny)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestPath", u8"TestScenes/helloBunny.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracerTests, Dragon)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestPath", u8"TestScenes/helloDragon.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracerTests, ReflectRefract)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestPath", u8"TestScenes/cornellGlass.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracerTests, Sponza)
{
    EnableVTMode();
    SimpleTracerSetup setup("TestPath", u8"TestScenes/crySponza.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}