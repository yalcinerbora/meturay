#include "MangledNames.h"

using namespace std::string_literals;

const std::string MangledNames::MaterialGroup(const char* tracerLogicName,
                                              const char* estimatorLogicName,
                                              const char* materialGroupName)
{
    std::string result = "(T)"s + tracerLogicName +
                         "(E)"s + estimatorLogicName +
                         "(M)"s + materialGroupName;
    return result;
}

const std::string MangledNames::MaterialBatch(const char* tracerLogicName,
                                              const char* estimatorLogicName,
                                              const char* primitiveGroupName,
                                              const char* materialGroupName)
{
    std::string result = "(T)"s + tracerLogicName +
                         "(E)"s + estimatorLogicName +
                         "(P)"s + primitiveGroupName +
                         "(M)"s + materialGroupName;
    return result;
}

const std::string MangledNames::AcceleratorGroup(const char* primitiveGroupName,
                                                 const char* acceleratorGroupName)
{
    std::string result =  "(P)"s + primitiveGroupName +
                          "(A)"s + acceleratorGroupName;
    return result;
}