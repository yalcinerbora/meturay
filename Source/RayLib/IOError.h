#pragma once
/**

I-O error "Enumeration"

*/

#include "Error.h"

struct IOError : public ErrorI
{
    public:
        enum Type
        {
            OK,
            // General
            FILE_NOT_FOUND,
            // Scene
            SCENE_CORRUPTED,
            // Maya nCache
            NCACHE_XML_ERROR,
            NCACHE_INVALID_FOURCC,
            NCACHE_INVALID_FORMAT,
            // Maya nCache Navier-Stokes Fluid
            NCACHE_DENSITY_NOT_FOUND,
            NCACHE_VELOCITY_NOT_FOUND,
            // Scene Json
            //....

            // End
            END
        };

    private:
        Type    type;

    public:
        // Constructors & Destructor
                        IOError() = default;
                        IOError(Type);
                        ~IOError() = default;

        operator        Type() const { return type; }
        operator        std::string() const override;
};

inline IOError::IOError(IOError::Type t)
    : type(t)
{}

inline IOError::operator std::string() const
{
    const char* const ErrorStrings[] =
    {
        "OK.",
        // General
        "File not found.",
        // Scene
        "Scene file is corrupted.",
        // Maya nCache
        "nCache XML parse error.",
        "nCache invalid fourcc code.",
        "nCache invalid file format code.",
        // Maya nCache Navier-Stokes Fluid
        "nCache \"density\" channel not found.",
        "nCache \"velocity\" channel not found."
    };
    static_assert((sizeof(ErrorStrings) / sizeof(const char*)) == static_cast<size_t>(IOError::END),
                  "Enum and enum string list size mismatch.");

    return ErrorStrings[static_cast<int>(type)];
}