#pragma once

/**

Tracer Node Interface.

This interface defines process behavior between

*/

#include "TracerCallbacksI.h"
#include "DistributorI.h"

class TracerNode
    : public TracerCallbacksI
    , public DistributorI
{
};