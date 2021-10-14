#include "MangledNames.h"

using namespace std::string_literals;

const std::string MangledNames::WorkBatch(const char* primitiveGroupName,
                                          const char* materialGroupName)
{
    std::string result = "(P)"s + primitiveGroupName +
                         "(M)"s + materialGroupName;
    return result;
}

const std::string MangledNames::BoundaryWorkBatch(const char* endpointGroupName)
{
    std::string result = "(E)"s + endpointGroupName;
    return result;
}

const std::string MangledNames::AcceleratorGroup(const char* primitiveGroupName,
                                                 const char* acceleratorGroupName)
{
    std::string result =  "(P)"s + primitiveGroupName +
                          "(A)"s + acceleratorGroupName;
    return result;
}

const std::string MangledNames::CameraGroup(const char* cameraGroupName)
{
    return "(C)"s + cameraGroupName;
}

const std::string MangledNames::LightGroup(const char* lightGroupName)
{
    return "(L)"s + lightGroupName;
}