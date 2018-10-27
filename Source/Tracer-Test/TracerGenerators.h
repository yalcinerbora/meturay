#pragma once

#include "TracerLib/TracerLogicGenerator.h"

class BasicTracerLogicGenerator final : public TracerLogicGenerator
{
	private:
		GPUTracerGen		tracerGenerator;
		GPUBaseAccelGen		baseAccelGenerator;

		GPUTracerPtr		tracerLogic;
		GPUBaseAccelPtr		baseAccelerator;
		
		

	protected:
	public:
		// Constructors & Destructor
							BasicTracerLogicGenerator();
							~BasicTracerLogicGenerator() = default;

		// Base Accelerator should be fetched after all the stuff is generated
		SceneError			GetBaseAccelerator(const std::string& accelType) override;
		// Finally get the tracer logic
		// Tracer logic will be constructed with respect to
		// Constructed batches
		SceneError			GetBaseLogic(TracerBaseLogicI*&) override;
};