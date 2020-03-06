#include "MangledNames.h"

using namespace std::string_literals;

const std::string MangledNames::WorkBatch(const char* primitiveGroupName,
                                          const char* materialGroupName)
{
    std::string result = "(P)"s + primitiveGroupName +
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