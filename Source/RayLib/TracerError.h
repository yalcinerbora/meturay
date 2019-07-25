#pragma once
/**

Tracer error "Enumeration"

*/

#include "Error.h"

struct TracerError : public ErrorI
{
    public:
        enum Type
        {
            OK,
            // Logical
            NO_LOGIC_SET,
            // General
            OUT_OF_MEMORY,
            // ...



            // End
            END
        };

    private:
        Type        type;

    public:
        // Constructors & Destructor
                    TracerError(Type);
                    ~TracerError() = default;

        operator    Type() const;
        operator    std::string() const override;
};

inline TracerError::TracerError(TracerError::Type t)
    : type(t)
{}

inline TracerError::operator Type() const
{
    return type;
}

inline TracerError::operator std::string() const
{
    static constexpr char const* const ErrorStrings[] =
    {
        "OK",
        "No Tracer Logic is set",
        // General
        "Out of Memory"
    };
    static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(TracerError::END),
                  "Enum and enum string list size mismatch.");

    return ErrorStrings[static_cast<int>(type)];
}