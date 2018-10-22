#include "TracerLoader.h"

LogicInterface TracerLoader::LoadTracerLogic(SharedLib& s)
{
	return LoadTracerLogic(s,
						   BaseInterfaceConstructorName,
						   BaseInterfaceDestructorName,
						   LogicInterfaceConstructorName,
						   LogicInterfaceDestructorName);
}

LogicInterface TracerLoader::LoadTracerLogic(SharedLib& s,
											 const char* baseConst,
											 const char* baseDest,
											 const char* logicConst,
											 const char* logicDest)
{
	// Load DLL	
	SharedLibPtr<TracerLogicGeneratorI> logicGen = s.GenerateObject<TracerLogicGeneratorI>(logicConst,
																						   logicDest);

	return logicGen;
}

TracerThread TracerLoader::GenerateTracerThread(TracerI& base, 
												TracerBaseLogicI& logic,
												uint32_t seed)
{
	return TracerThread(base, logic, seed);
}