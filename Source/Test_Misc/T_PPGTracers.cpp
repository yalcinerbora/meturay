#include "SimpleTracerSetup.h"

TEST(PPGTracers, CornellBox)
{
    EnableVTMode();
    SimpleTracerSetup setup("PPGTracer", true,
                            u8"TestScenes/helloCornell.json", 0.0);
    ASSERT_TRUE(setup.Init());
    setup.Body();
}