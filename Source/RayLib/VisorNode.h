#pragma once
/**

V

This Distributor is user interfacable thorough
Visor and Analytic Classes/Programs.

This delegates user input and receives user output (image)

*/

#include "VisorCallbacksI.h"
#include "DistributorI.h"

class VisorNodeI
    : public VisorCallbacksI
    , public DistributorI
{
};