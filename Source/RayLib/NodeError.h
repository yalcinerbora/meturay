#pragma once

#pragma once
/**

Tracer error "Enumeration"

*/

#include "Error.h"

struct NodeError : public ErrorI
{
    public:
        enum Type
        {
            OK,
            // Connection
            CONNECTION_FAILED,
            // ...

            // End
            END
        };

    private:
        Type        type;

    public:
        // Constructors & Destructor
                    NodeError(Type);
                    ~NodeError() = default;

        operator    Type() const;
        operator    std::string() const override;
};

inline NodeError::NodeError(NodeError::Type t)
    : type(t)
{}

inline NodeError::operator Type() const
{
    return type;
}

inline NodeError::operator std::string() const
{
    static constexpr char const* const ErrorStrings[] =
    {
        "OK",
        "Connection cannot be established"
    };
    static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(NodeError::END),
                  "Enum and enum string list size mismatch.");

    return ErrorStrings[static_cast<int>(type)];
}