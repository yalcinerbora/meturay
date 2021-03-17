#include "SimpleTracerSetup.h"

TEST(AOTracers, CornellBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("AOTracer", u8"TestScenes/helloCornell.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}