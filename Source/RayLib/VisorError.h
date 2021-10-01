#pragma once

#pragma once
/**

Tracer error "Enumeration"

*/

#include "Error.h"
#include <stdexcept>

struct VisorError : public ErrorI
{
    public:
        enum Type
        {
            OK,

            WINDOW_GENERATOR_ERROR,
            WINDOW_GENERATION_ERROR,
            RENDER_FUCTION_GENERATOR_ERROR,

            // Image Related
            IMAGE_IO_ERROR,

            // Guide Debug Related
            NO_LOGIC_FOR_GUIDE_DEBUGGER,

            // End
            END
        };

    private:
        Type        type;

    public:
        // Constructors & Destructor
                    VisorError(Type);
                    ~VisorError() = default;

        operator    Type() const;
        operator    std::string() const override;
};

class VisorException : public std::runtime_error
{
    private:
        VisorError          e;

    protected:
    public:
        VisorException(VisorError::Type t)
            : std::runtime_error("")
            , e(t)
        {}
        VisorException(VisorError::Type t, const char* const err)
            : std::runtime_error(err)
            , e(t)
        {}
        operator VisorError() const { return e; };
};

inline VisorError::VisorError(VisorError::Type t)
    : type(t)
{}

inline VisorError::operator Type() const
{
    return type;
}

inline VisorError::operator std::string() const
{
    static constexpr char const* const ErrorStrings[] =
    {
        "OK",
        "Window generator failed to initialize",
        "Window generator is unable to generate window",
        "Render function generator failed to initialize",
        // Image Related
        "ImageIO Error",
        // Guide Debug Related
        "No logic found for that guide debugger type"
    };
    static_assert(std::extent<decltype(ErrorStrings)>::value == static_cast<size_t>(VisorError::END),
                  "Enum and enum string list size mismatch.");

    return ErrorStrings[static_cast<int>(type)];
}