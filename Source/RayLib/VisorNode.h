#pragma once
/**

V

This Distributor is user interfacable thorugh
Visor and Analytic Classes/Programs.

This deleagtes user input and recieves user output (image)

*/

#include "VisorCallbacksI.h"
#include "DistributorI.h"

class VisorNodeI
	: public VisorCallbacksI
	, public DistributorI
{

};