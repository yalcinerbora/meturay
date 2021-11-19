#include "SimpleTracerSetup.h"

TEST(AOTracers, CornellBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("AOTracer", u8"TestScenes/helloCornell.json", 0.0,
                            false);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}