#include "SimpleTracerSetup.h"

TEST(PPGTracers, CornellBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("PathTracer", true,
                            u8"TestScenes/helloCornell.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}