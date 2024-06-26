#pragma once

#include <string>

namespace MangledNames
{
    const std::string WorkBatch(const char* primitiveGroupName,
                                const char* materialGroupName);
    const std::string BoundaryWorkBatch(const char* endpointGroupName);

    const std::string AcceleratorGroup(const char* primitiveGroupName,
                                       const char* acceleratorGroupName);
    const std::string CameraGroup(const char* cameraGroupName);
    const std::string LightGroup(const char* lightGroupName);
}

#define TYPENAME_DEF(Function, name)\
static const char* TypeName()\
{\
    static const std::string typeName = MangledNames::Function(name);\
    return typeName.c_str();\
}

#define ACCELERATOR_TYPE_NAME(name, P)\
public: static const char* TypeName()\
{\
    static std::string typeName = MangledNames::AcceleratorGroup(P::TypeName(),\
                                                                 name);\
    return typeName.c_str();\
}
