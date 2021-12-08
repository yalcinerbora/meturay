#pragma once
#pragma once
/**

DLL Error "Enumeration"

*/

#include "Error.h"
#include <stdexcept>

struct DLLError : public ErrorI
{
    public:
        enum Type
        {
            OK,
            // Logical
            DLL_NOT_FOUND,
            MANGLED_NAME_NOT_FOUND,
            // End
            END
        };

    private:
        Type        type;

    public:
                    // Constructors & Destructor
                    DLLError(Type);
                    ~DLLError() = default;

                    operator Type() const;
                    operator std::string() const override;
};

class DLLException : public std::runtime_error
{
    private:
        DLLError        e;

    protected:
    public:
                        DLLException(DLLError::Type t)
                            : std::runtime_error("")
                            , e(t)
                        {}
                        DLLException(DLLError::Type t, const char* const err)
                            : std::runtime_error(err)
                            , e(t)
                        {}
                        operator DLLError() const { return e; };
};

inline DLLError::DLLError(DLLError::Type t)
    : type(t)
{}

inline DLLError::operator Type() const
{
    return type;
}

inline DLLError::operator std::string() const
{
    static constexpr char const* const ErrorStrings[] =
    {
        "OK",
        // Logical
        "Shared Library not found",
        "Unable to find shared function"
    };
    static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(DLLError::END),
                  "Enum and enum string list size mismatch.");

    return ErrorStrings[static_cast<int>(type)];
}