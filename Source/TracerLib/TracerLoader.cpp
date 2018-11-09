#include "TracerLoader.h"

LogicInterface TracerLoader::LoadTracerLogic(SharedLib& s)
{
	return LoadTracerLogic(s,
						   BaseInterfaceConstructorName,
						   BaseInterfaceDestructorName);
}

LogicInterface TracerLoader::LoadTracerLogic(SharedLib& s,
											 const char* generatorConstructor,
											 const char* generatorDestructor)
{
	// Load DLL	
	SharedLibPtr<TracerLogicGeneratorI> logicGen = s.GenerateObject<TracerLogicGeneratorI>(generatorConstructor,
																						   generatorDestructor);

	return logicGen;
}

TracerThread TracerLoader::GenerateTracerThread(TracerI& base, 
												TracerBaseLogicI& logic,
												uint32_t seed)
{
	return TracerThread(base, logic, seed);
}