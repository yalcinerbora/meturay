#include "TracerLoader.h"

LogicInterfaces TracerLoader::LoadTracerLogic(SharedLib& s)
{
	return LoadTracerLogic(s,
						   BaseInterfaceConstructorName,
						   BaseInterfaceDestructorName,
						   LogicInterfaceConstructorName,
						   LogicInterfaceDestructorName);
}

LogicInterfaces TracerLoader::LoadTracerLogic(SharedLib& s,
											  const char* baseConst,
											  const char* baseDest,
											  const char* logicConst,
											  const char* logicDest)
{
	// Load DLL
	SharedLibPtr<TracerBaseLogicI> baseLogic = s.GenerateObject<TracerBaseLogicI>(baseConst, baseDest);
	SharedLibPtr<TracerLogicGeneratorI> logicGen = s.GenerateObject<TracerLogicGeneratorI>(logicConst,
																						   logicDest);

	return std::make_pair(baseLogic, logicGen);
}

TracerThread TracerLoader::GenerateTracerThread(TracerBase& base, 
												TracerBaseLogicI& logic,
												uint32_t seed)
{
	return TracerThread(base, logic, seed);
}