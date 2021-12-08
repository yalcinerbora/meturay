#pragma once
/**
Basic Interface for Node System
nodes can be multiple and it can be joined to node packs

Nodes have single purpose (Except Self-Node):
 - Trace scene (Tracer Node)
 - Show current progress of tracing (Visor Node)
 - Change options, do actions (Command Node)
 - Show analytic data (Analytic Node)

Self node does everything (for workstation use- or debug use etc.)

*/

#include "NodeError.h"
//#include "TracerStatus.h"

class NodeI
{
    public:
        virtual                     ~NodeI() = default;

        // Interface
        virtual NodeError           Initialize() = 0;
//        virtual TracerStatus        Status() const = 0;

        // Main Thread
        virtual void                Work() = 0;
};