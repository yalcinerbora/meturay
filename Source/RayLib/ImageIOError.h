#pragma once

#include "Error.h"

struct ImageIOError : public ErrorI
{
    public:
        enum Type
        {
            OK,
            // Common
            IMAGE_NOT_FOUND,            
            UNKNOWN_PIXEL_FORMAT,
            UNKNOWN_IMAGE_TYPE,
            // Conversion
            TEMPLATE_TYPE_IS_NOT_COMPATIBLE,
            FORMATS_ARE_NOT_CONVERTIBLE,
            // Internal Errors
            READ_INTERNAL_ERROR,

            END
        };

    private:
        Type        type;
        std::string extra;

    public:
        // Constructors & Destructor
                    ImageIOError(Type);
                    ImageIOError(Type, const std::string& extra);
                    ~ImageIOError() = default;

        operator    Type() const;
        operator    std::string() const override;
};

class ImageIOException : public std::runtime_error
{
    private:
        ImageIOError        e;

    protected:
    public:
        ImageIOException(ImageIOError e)
            : std::runtime_error("")
            , e(e)
        {}
        ImageIOException(ImageIOError e, const std::string& s)
            : std::runtime_error("")
            , e(e, s)
        {}
        ImageIOException(ImageIOError::Type t, const char* const err)
            : std::runtime_error(err)
            , e(t)
        {}
        operator ImageIOError() const { return e; };
};

inline ImageIOError::ImageIOError(Type t)
    : type(t)
{}

inline ImageIOError::ImageIOError(Type t, const std::string& extra)
    : type(t)
    , extra(extra)
{}

inline ImageIOError::operator Type() const
{
    return type;
}

inline ImageIOError::operator std::string() const
{
    static constexpr char const* const ErrorStrings[] =
    {
        "OK",
        // Common
        "Image file not found",
        "Unknown pixel format",
        "Unknown image type",
        // Conversion
        "Template type is not compatible wrt the compatible format",
        "Image format and requested format are not convertible",
        // Internal Errors
        "Read operation internal error"
    };
    static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(ImageIOError::END),
                  "Enum and enum string list size mismatch.");

    if(extra.empty())
        return ErrorStrings[static_cast<int>(type)];
    else
        return std::string(ErrorStrings[static_cast<int>(type)]) + " (" + extra + ')';
}