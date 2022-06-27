#pragma once
/*
    Implementation of Tracer options using std map and a variant

    This is a memory implementation
    There may be file implementation (like json etc)
*/

#include <variant>
#include <map>
#include <string>
#include <nlohmann/json_fwd.hpp>

#include "OptionsI.h"

// ORDER OF THIS SHOULD BE SAME AS THE "OPTION_TYPE" ENUM
using OptionVariable = std::variant<bool, int64_t, float,
                                    Vector2l,
                                    Vector2, Vector3, Vector4,
                                    std::string>;
// Cuda (nvcc) did not liked this :(
using VariableList = std::map<std::string, OptionVariable>;

class Options : public OptionsI
{
    private:
        VariableList    variables;

        template <class T>
        TracerError     Get(T&, const std::string&) const;
        template <class T>
        TracerError     Set(const T&, const std::string&);

    public:
        // Constructors & Destructor
                        Options();
                        Options(VariableList&&);
                        Options(const nlohmann::json&);
                        ~Options() = default;

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

inline Options::Options()
{}

inline Options::Options(VariableList&& v)
 : variables(v)
{}

template <class T>
TracerError Options::Get(T& v, const std::string& s) const
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
TracerError Options::Set(const T& v, const std::string& s)
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

inline TracerError Options::GetType(OptionType& t, const std::string& s) const
{
    auto loc = variables.end();
    if((loc = variables.find(s)) == variables.end())
    {
        return TracerError::OPTION_NOT_FOUND;
    }
    t = static_cast<OptionType>(loc->second.index());
    return TracerError::OK;
}

inline TracerError Options::GetBool(bool& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError Options::GetString(std::string& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError Options::GetFloat(float& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError Options::GetVector2(Vector2& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError Options::GetVector3(Vector3& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError Options::GetVector4(Vector4& v, const std::string& s) const
{
    return Get(v, s);
}

inline TracerError Options::GetInt(int32_t& v, const std::string& s) const
{
    int64_t val;
    TracerError e = TracerError::OK;

    if((e = Get(val, s)) != TracerError::OK)
       return e;

    if(val < std::numeric_limits<int32_t>::min() ||
       val > std::numeric_limits<int32_t>::max())
        return TracerError::OPTION_TYPE_MISMATCH;

    v = static_cast<int32_t>(val);
    return e;
}

inline TracerError Options::GetUInt(uint32_t& v, const std::string& s) const
{
    int64_t val;
    TracerError e = TracerError::OK;

    if((e = Get(val, s)) != TracerError::OK)
       return e;

    if(val < std::numeric_limits<uint32_t>::min() ||
       val > std::numeric_limits<uint32_t>::max())
        return TracerError::OPTION_TYPE_MISMATCH;

    v = static_cast<uint32_t>(val);
    return e;
}

inline TracerError Options::GetVector2i(Vector2i& v, const std::string& s) const
{
    Vector2l val;
    TracerError e = TracerError::OK;

    if((e = Get(val, s)) != TracerError::OK)
        return e;

    if(val < Vector2l(std::numeric_limits<int32_t>::min()) ||
       val > Vector2l(std::numeric_limits<int32_t>::max()))
        return TracerError::OPTION_TYPE_MISMATCH;

    v = Vector2i(val[0], val[1]);
    return e;
}

inline TracerError Options::GetVector2ui(Vector2ui& v, const std::string& s) const
{
    Vector2l val;
    TracerError e = TracerError::OK;

    if((e = Get(val, s)) != TracerError::OK)
        return e;

    if(val < Vector2l(std::numeric_limits<uint32_t>::min()) ||
       val > Vector2l(std::numeric_limits<uint32_t>::max()))
        return TracerError::OPTION_TYPE_MISMATCH;

    v = Vector2ui(val[0], val[1]);
    return e;
}
//==================================
inline TracerError Options::SetBool(bool v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError Options::SetString(const std::string& v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError Options::SetFloat(float v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError Options::SetVector2(const Vector2& v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError Options::SetVector3(const Vector3& v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError Options::SetVector4(const Vector4& v, const std::string& s)
{
    return Set(v, s);
}

inline TracerError Options::SetInt(int32_t v, const std::string& s)
{
    return Set(static_cast<int64_t>(v), s);
}

inline TracerError Options::SetUInt(uint32_t v, const std::string& s)
{
    return Set(static_cast<int64_t>(v), s);
}

inline TracerError Options::SetVector2i(const Vector2i v, const std::string& s)
{
    return Set(Vector2l(v), s);
}

inline TracerError Options::SetVector2ui(const Vector2ui v, const std::string& s)
{
    return Set(Vector2l(v), s);
}
