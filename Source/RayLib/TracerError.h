#pragma once
/**

Tracer error "Enumeration"

*/

#include "Error.h"
#include <stdexcept>

struct TracerError : public ErrorI
{
    public:
        enum Type
        {
            OK,
            // Logical
            NO_LOGIC_SET,
            // General
            CPU_OUT_OF_MEMORY,
            GPU_OUT_OF_MEMORY,
            // Options Related
            UNKNOWN_OPTION,
            OPTION_TYPE_MISMATCH,
            // Accelerator Related
            UNABLE_TO_CONSTRUCT_BASE_ACCELERATOR,
            UNABLE_TO_CONSTRUCT_ACCELERATOR,
            UNABLE_TO_CONSTRUCT_CAMERA,
            UNABLE_TO_CONSTRUCT_LIGHT,
            UNABLE_TO_CONSTRUCT_TEXTURE_REFERENCE,
            // Work Related
            UNABLE_TO_GENERATE_WORK,
            // Initialization Related
            UNABLE_TO_INITIALIZE,
            UNKNOWN_SCENE_PARTITIONER_TYPE,
            NO_LOGIC_FOR_TRACER,
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

class TracerException : public std::runtime_error
{
    private:
        TracerError          e;

    protected:
    public:
        TracerException(TracerError::Type t)
            : std::runtime_error("")
            , e(t)
        {}
        TracerException(TracerError::Type t, const char* const err)
            : std::runtime_error(err)
            , e(t)
        {}
        operator TracerError() const { return e; };
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
        "CPU is out of memory",
        "GPU is out of memory",
        // Option Related
        "Unknown option",
        "Option type mismatch",
        // Accelerator Related
        "Unable to construct base accelerator",
        "Unable to construct accelerator",
        "Unable to construct camera",
        "Unable to construct light",
        "Unable to construct texture reference",
        // Work Related
        "Unable to generate work for material/primitive pair",
        // Initialization Related
        "Unable to initialize tracer",
        "Unknown scene partitioner type",
        "No logic found for that tracer"
    };
    static_assert(std::extent<decltype(ErrorStrings)>::value == static_cast<size_t>(TracerError::END),
                  "Enum and enum string list size mismatch.");

    return ErrorStrings[static_cast<int>(type)];
}