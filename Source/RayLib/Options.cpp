#include "Options.h"
#include "SceneIO.h"
#include <nlohmann/json.hpp>

#include "RayLib/Vector.h"

Options::Options(const nlohmann::json& jsonObj)
{
    for(const auto& optsNode : jsonObj.items())
    {
        std::string name = optsNode.key();
        const auto& val = optsNode.value();

        OptionVariable v;
        if(val.is_string())
        {
            v = val.get<std::string>();
        }
        else if(val.is_boolean())
        {
            v = val.get<bool>();
        }
        else if(val.is_number_float())
        {
            v = val.get<float>();
        }
        else if(val.is_number_integer())
        {
            v = val.get<int64_t>();
        }
        // Array type; may be vector etc.
        else if(val.is_array())
        {
            //const auto& a = val.array;
            if(val[0].is_number_integer())
            {
                if(val.size() != 2)
                    throw std::runtime_error("Invalid Options Type");

                auto arr = val.get<std::array<int64_t, 2>>();
                v = Vector<2, int64_t>(arr.data());
            }
            else if(val[0].is_number_float())
            {
                if(val.size() == 1 || val.size() > 4)
                    throw std::runtime_error("Invalid Options Type");
                else if(val.size() == 2)
                {
                    auto arr = val.get<std::array<float, 2>>();
                    v = Vector<2, float>(arr.data());
                }
                else if(val.size() == 3)
                {
                    auto arr = val.get<std::array<float, 3>>();
                    v = Vector<3, float>(arr.data());
                }
                else if(val.size() == 4)
                {
                    auto arr = val.get<std::array<float, 4>>();
                    v = Vector<4, float>(arr.data());
                }
            }
            else throw std::runtime_error("Invalid Options Type");
        }
        // Other types are invalid for opts
        else throw std::runtime_error("Invalid Options Type");

        variables.emplace(name, v);
    }
}