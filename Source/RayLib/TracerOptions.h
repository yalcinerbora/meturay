#pragma once
/*
    Implementation of Tracer options using std map and a variant

    This is a memory implementation
    There may be file implementation (like json etc)
*/

#include <variant>
#include <map>
#include <string>

#include "TracerOptionsI.h"

// ORDER OF THIS SHOULD BE SAME AS THE "OPTION_TYPE" ENUM
using OptionVariable = std::variant<bool, int32_t, uint32_t, float,
                                    Vector2i, Vector2ui,
                                    Vector2, Vector3, Vector4,
                                    std::string>;
// Cuda (nvcc) did not liked this :(

using VariableList = std::map<std::string, OptionVariable>;

class TracerOptions : public TracerOptionsI
{
    private:
        VariableList    variables;

        template <class T>
        TracerError     Get(T&, const std::string&) const;
        template <class T>
        TracerError     Set(const T&, const std::string&);

    public:
        // Constructors & Destructor
                        TracerOptions();
                        TracerOptions(VariableList&&);
                        ~TracerOptions() = default;

        // Interface
        TracerError     GetType(OptionType&, const std::string&) const override;
        //
        TracerError     GetBool(bool&, const std::string&) const override;

        TracerError     GetString(std::string&, const std::string&) const override;

        TracerError     GetFloat(float&, const std::string&) const override;
        TracerError     GetVector2(Vector2&, const std::string&) const override;
        TracerError     GetVector3(Vector3&, const std::string&) const override;
        TracerError     GetVector4(Vector4&, const std::string&) const override;

        TracerError     GetInt(int32_t&, const std::string&) const override;
        TracerError     GetUInt(uint32_t&, const std::string&) const override;
        TracerError     GetVector2i(Vector2i&, const std::string&) const override;
        TracerError     GetVector2ui(Vector2ui&, const std::string&) const override;
        //
        TracerError     SetBool(bool, const std::string&) override;

        TracerError     SetString(const std::string&, const std::string&) override;

        TracerError     SetFloat(float, const std::string&) override;
        TracerError     SetVector2(const Vector2&, const std::string&) override;
        TracerError     SetVector3(const Vector3&, const std::string&) override;
        TracerError     SetVector4(const Vector4&, const std::string&) override;

        TracerError     SetInt(int32_t, const std::string&) override;
        TracerError     SetUInt(uint32_t, const std::string&) override;
        TracerError     SetVector2i(const Vector2i, const std::string&) override;
        TracerError     SetVector2ui(const Vector2ui, const std::string&) override;
};

inline TracerOptions::TracerOptions()
{}

inline TracerOptions::TracerOptions(VariableList&& v)
 : variables(v)
{}

template <class T>
TracerError TracerOptions::Get(T& v, const std::string& s) const
{
    auto loc = variables.end();
    if((loc = variables.find(s)) == variables.end())
    {
        return TracerError::OPTION_NOT_FOUND;
    }
    try { v = std::get<T>(loc->second); }
    catch(const std::bad_variant_access&) { return TracerError::OPTION_TYPE_MISMATCH; }
    return TracerError::OK;
}

template <class T>
TracerError TracerOptions::Set(const T& v, const std::string& s)
{
    auto loc = variables.end();
    if((loc = variables.find(s)) == variables.end())
    {
        return TracerError::OPTION_NOT_FOUND;
    }
    if(std::holds_alternative<T>(loc->second))
    {
        loc->second = v;
    }
    else return TracerError::OPTION_TYPE_MISMATCH;
    return TracerError::OK;
}

inline TracerError TracerOptions::GetType(OptionType& t, const std::string& s) const
{
    auto loc = variables.end();
    if((loc = variables.find(s)) == variables.end())
    {
        return TracerError::OPTION_NOT_FOUND;
    }
    t = static_cast<OptionType>(loc->second.index());
    return TracerError::OK;
}

inline TracerError TracerOptions::GetBool(bool& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError TracerOptions::GetString(std::string& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError TracerOptions::GetFloat(float& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError TracerOptions::GetVector2(Vector2& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError TracerOptions::GetVector3(Vector3& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError TracerOptions::GetVector4(Vector4& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError TracerOptions::GetInt(int32_t& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError TracerOptions::GetUInt(uint32_t& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError TracerOptions::GetVector2i(Vector2i& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError TracerOptions::GetVector2ui(Vector2ui& v, const std::string& s) const
{
    return Get(v, s);
}
//==================================
inline TracerError TracerOptions::SetBool(bool v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError TracerOptions::SetString(const std::string& v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError TracerOptions::SetFloat(float v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError TracerOptions::SetVector2(const Vector2& v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError TracerOptions::SetVector3(const Vector3& v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError TracerOptions::SetVector4(const Vector4& v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError TracerOptions::SetInt(int32_t v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError TracerOptions::SetUInt(uint32_t v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError TracerOptions::SetVector2i(const Vector2i v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError TracerOptions::SetVector2ui(const Vector2ui v, const std::string& s)
{
    return Set(v, s);
}
