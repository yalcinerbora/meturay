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

#define MATERIAL_TYPE_NAME(name, T, E) \
    public:\
    static const char* Name()\
    {\
    return ##name##; \
    }\
    static const char* TypeName()\
    {\
        static const std::string typeName = MangledNames::MaterialGroup(##T##::TypeName(),\
                                                                        ##E##::TypeName(),\
                                                                        ##name##);\
        return typeName.c_str();\
    }