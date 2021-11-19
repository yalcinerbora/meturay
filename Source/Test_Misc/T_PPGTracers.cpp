#include "SimpleTracerSetup.h"

TEST(PPGTracers, PPGDirection)
{
    EnableVTMode();
    SimpleTracerSetup setup("PPGTracer", u8"TestScenes/ppgTest.json", 0.0,
                            true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}

TEST(PPGTracers, CornellBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("PPGTracer", u8"TestScenes/helloCornell.json", 0.0,
                            true);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}