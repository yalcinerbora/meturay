#pragma once

/*
    Simple  (String - Type - Value) triplet interface

    each tracer will have many different options
*/

#include <string>

#include "Vector.h"

class TracerOptionsI
{
    public:
        enum  OptionType
        {
            BOOL,
            INT32,
            UINT32,
            FLOAT,
            VECTOR2I,
            VECTOR2UI,
            VECTOR2,
            VECTOR3,
            VECTOR4,
            STRING,

            END
        };

    public:
        virtual                 ~TracerOptionsI() = default;

        // Interface
        virtual TracerError     GetType(OptionType&, const std::string&) const = 0;
        //
        virtual TracerError     GetBool(bool&, const std::string&) const = 0;
        virtual TracerError     GetString(std::string&, const std::string&) const = 0;

        virtual TracerError     GetFloat(float&, const std::string&) const = 0;
        virtual TracerError     GetVector2(Vector2&, const std::string&) const = 0;
        virtual TracerError     GetVector3(Vector3&, const std::string&) const = 0;
        virtual TracerError     GetVector4(Vector4&, const std::string&) const = 0;

        virtual TracerError     GetInt(int32_t&, const std::string&) const = 0;
        virtual TracerError     GetUInt(uint32_t&, const std::string&) const = 0;
        virtual TracerError     GetVector2i(Vector2i&, const std::string&) const = 0;
        virtual TracerError     GetVector2ui(Vector2ui&, const std::string&) const = 0;
        //
        virtual TracerError     SetBool(bool, const std::string&) = 0;

        virtual TracerError     SetString(const std::string&, const std::string&) = 0;

        virtual TracerError     SetFloat(float, const std::string&) = 0;
        virtual TracerError     SetVector2(const Vector2&, const std::string&) = 0;
        virtual TracerError     SetVector3(const Vector3&, const std::string&) = 0;
        virtual TracerError     SetVector4(const Vector4&, const std::string&) = 0;

        virtual TracerError     SetInt(int32_t, const std::string&) = 0;
        virtual TracerError     SetUInt(uint32_t, const std::string&) = 0;
        virtual TracerError     SetVector2i(const Vector2i, const std::string&) = 0;
        virtual TracerError     SetVector2ui(const Vector2ui, const std::string&) = 0;
};
