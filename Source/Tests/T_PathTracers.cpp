#include "SimpleTracerSetup.h"

TEST(PathTracers, CornellBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", true,
                            u8"TestScenes/helloCornell.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, Bunny)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", true,
                            u8"TestScenes/helloBunny.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, Dragon)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", true,
                            u8"TestScenes/helloDragon.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, ReflectRefract)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", true,
                            u8"TestScenes/cornellGlass.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, SkySphereHDR)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", true,
                            u8"TestScenes/skySphereHDR.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, Door)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", true,
                            u8"Scenes/veachDoor.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, HDRReflection)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", true,
                            u8"TestScenes/hdrReflection.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, HDRRefraction)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", true,
                            u8"TestScenes/hdrRefraction.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

//TEST(PathTracers, Sponza)
//{
//    ////EnableVTMode();
//    ////SimpleTracerSetup setup("TestPath", u8"Scenes/crySponza.json", 0.0);
//    ////ASSERT_TRUE(setup.Init());
//    ////setup.Body();
//}