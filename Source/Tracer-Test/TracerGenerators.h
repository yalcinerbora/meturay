#pragma once

#include "TracerLib/TracerLogicGenerator.h"

class BasicTracerLogicGenerator final : public TracerLogicGenerator
{
	private:
		GPUTracerGen		tracerGenerator;
		GPUTracerPtr		tracerLogic;

	protected:
	public:
		// Constructors & Destructor
							BasicTracerLogicGenerator();
							~BasicTracerLogicGenerator() = default;

		// Finally get the tracer logic
		// Tracer logic will be constructed with respect to
		// Constructed batches
		SceneError			GetBaseLogic(TracerBaseLogicI*&) override;
};