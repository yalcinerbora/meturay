#pragma once

/**

Functionalty to Load DLLs or SOs


*/

#include <string>
#include <memory>

#include "ObjectFuncDefinitions.h"

class SharedLib
{
	private:
		static constexpr const char* WinDLLExt		= ".dll";
		static constexpr const char* LinuxDLLExt	= ".so";

		// Props
		void*				libHandle;

		// Internal
		void*				GetProcAdressInternal(const std::string& fName);

	protected:
	public:
		// Constructors & Destructor
							SharedLib(const std::string& libName);
							SharedLib(const SharedLib&) = delete;
		SharedLib&			operator=(const SharedLib&) = delete;
							~SharedLib();

	template <class T>
	SharedLibPtr<T>			GenerateObject(const std::string& mangledConstructorName,
										   const std::string& mangledDestructorName);
};

template <class T>
SharedLibPtr<T> SharedLib::GenerateObject(const std::string& mangledConstructorName,
										  const std::string& mangledDestructorName)
{
	ObjGeneratorFunc<T> genFunc = GetProcAdress(mangledConstructorName);
	ObjDestroyerFunc<T> destFunc = GetProcAdress(mangledDestructorName);

	return std::make_unqiue(ObjGeneratorFunc(), *destFunc);
}