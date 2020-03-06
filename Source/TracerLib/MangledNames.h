#pragma once

#include <string>

namespace MangledNames
{
    const std::string WorkBatch(const char* primitiveGroupName,
                                const char* materialGroupName);
    const std::string AcceleratorGroup(const char* primitiveGroupName,
                                       const char* acceleratorGroupName);
}

#define ACCELERATOR_TYPE_NAME(name, P)\
public: static const char* TypeName()\
{\
    static std::string typeName = MangledNames::AcceleratorGroup(##P##::TypeName(),\
                                                                 ##name##);\
    return typeName.c_str();\
}