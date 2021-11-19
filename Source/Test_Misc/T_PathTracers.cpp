#include "SimpleTracerSetup.h"

TEST(PathTracers, CornellBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer",  u8"TestScenes/helloCornell.json", 0.0,
                            true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, Bunny)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", u8"TestScenes/helloBunny.json", 0.0,
                            true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, Dragon)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", u8"TestScenes/helloDragon.json", 0.0,
                            true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, ReflectRefract)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", u8"TestScenes/cornellGlass.json", 0.0,
                            true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, HDRReflection)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", u8"TestScenes/hdrReflection.json", 0.0,
                            true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, HDRRefraction)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", u8"TestScenes/hdrRefraction.json", 0.0,
                            true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, NormalMap)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", u8"TestScenes/normalMap.json", 0.0,
                            true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PathTracers, UnrealMaterial)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", u8"TestScenes/unrealMaterial.json", 0.0,
                            true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}