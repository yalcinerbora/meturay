
# Compile options for the entire system
# Structure is inspired by the
# https://github.com/cpp-best-practices/cmake_template/blob/main/cmake/CompilerWarnings.cmake
# I did not know you can add multiple expression on the generator expressions
# i.e:
#   $<$<COMPILE_LANGUAGE:CXX>:/W3>
#   $<$<COMPILE_LANGUAGE:CXX>:/Zi>
#
#   $<$<COMPILE_LANGUAGE:CXX>:/Zi;W3>

# Create Scope
block(SCOPE_FOR VARIABLES)

#===============#
#    COMPILE    #
#===============#
# Warnings
set(MRAY_MSVC_OPTIONS

    # From Json Turner
    /W4             # Baseline reasonable warnings
    /w14242         # 'identifier': conversion from 'type1' to 'type1', possible loss of data
    /w14254         # 'operator': conversion from 'type1:field_bits' to 'type2:field_bits', possible loss of data
    /w14263         # 'function': member function does not override any base class virtual member function
    /w14265         # 'classname': class has virtual functions, but destructor is not virtual instances of this class may not
                    # be destructed correctly
    /w14287         # 'operator': unsigned/negative constant mismatch
    /we4289         # nonstandard extension used: 'variable': loop control variable declared in the for-loop is used outside
                    # the for-loop scope
    /w14296         # 'operator': expression is always 'boolean_value'
    /w14311         # 'variable': pointer truncation from 'type1' to 'type2'
    /w14545         # expression before comma evaluates to a function which is missing an argument list
    /w14546         # function call before comma missing argument list
    /w14547         # 'operator': operator before comma has no effect; expected operator with side-effect
    /w14549         # 'operator': operator before comma has no effect; did you intend 'operator'?
    /w14555         # expression has no effect; expected expression with side- effect
    /w14619         # pragma warning: there is no warning number 'number'
    /w14640         # Enable warning on thread un-safe static member initialization
    /w14826         # Conversion from 'type1' to 'type_2' is sign-extended. This may cause unexpected runtime behavior.
    /w14905         # wide string literal cast to 'LPSTR'
    /w14906         # string literal cast to 'LPWSTR'
    /w14928         # illegal copy-initialization; more than one user-defined conversion has been implicitly applied
    /permissive-    # standards conformance mode for MSVC compiler.

    /external:W0    # Minimize warnings on external stuff
                    # i.e. it is included with the <...> syntax.

    # TODO: Reason about the Debug and Release Builds
    #/Z7             # Internally put the PDB files (for debugging)
)

set(MRAY_CLANG_OPTIONS
    -Wall
    -Wextra                 # reasonable and standard
    -Wshadow                # warn the user if a variable declaration shadows one from a parent context
    -Wnon-virtual-dtor      # warn the user if a class with virtual functions has a non-virtual destructor. This helps
                            # catch hard to track down memory errors
    -Wold-style-cast        # warn for c-style casts
    -Wcast-align            # warn for potential performance problem casts
    -Wunused                # warn on anything being unused
    -Woverloaded-virtual    # warn if you overload (not override) a virtual function
    -Wpedantic              # warn if non-standard C++ is used
    -Wconversion            # warn on type conversions that may lose data
    -Wsign-conversion       # warn on sign conversions
    -Wnull-dereference      # warn if a null dereference is detected
    -Wdouble-promotion      # warn if float is implicit promoted to double
    -Wformat=2              # warn on security issues around functions that format output (ie printf)
    -Wimplicit-fallthrough  # warn on statements that fallthrough without an explicit annotation
)

set(MRAY_GCC_OPTIONS
    ${CLANG_WARNINGS}
    -Wmisleading-indentation    # warn if indentation implies blocks where blocks do not exist
    -Wduplicated-cond           # warn if if / else chain has duplicated conditions
    -Wduplicated-branches       # warn if if / else branches have duplicated code
    -Wlogical-op                # warn about logical operations being used where bitwise were probably wanted
    -Wuseless-cast              # warn if you perform a cast to the same type
)

set(MRAY_CUDA_OPTIONS
    # From Jason Turner
    -Wall
    -Wextra
    -Wunused
    -Wconversion
    -Wshadow
    # Debug Related
    $<$<CONFIG:Debug>:-G
    $<$<CONFIG:Release>:-lineinfo

    # Extended Lambdas (__device__ tagged lambdas)
    -extended-lambda

    # Use fast math creates self intersection issues on
    # some scenes. Thus, we only selectively enable
    # some fp optimizations (culprit was prec-sqrt on some Unreal Materials)
    -use_fast_math
    # Sometimes fast math creates issues, these are commented
    # here to select the problem causing flag if an error occurs
    #--fmad=true
    #--ftz=true
    #--prec-sqrt=false
    #--prec-div=false

    # Misc.
    -restrict
    -extra-device-vectorization
)

# Platform Specific CUDA Options
if(MSVC)
    set(CUDA_WARNINGS ${CUDA_WARNINGS}
        # Ignore inline functions are not defined warning
        -Xcompiler=/wd4984
        -Xcompiler=/wd4506
        -Xcompiler=/W3
        -Xcompiler=/Zi
        -Xcompiler=/external:W0
        # Test
        # DROPPED W4 On MSVC it shows many unnecessary info
        # These Flags for W4 however /external does not work properly i think
        # it still returns warnings and clutters the code extensively
        # Thus these are commented out
    )
else()
    set(CUDA_WARNINGS ${CUDA_WARNINGS}
        -Xcompiler=-g3)
endif()

# Combine Depending on the platform
if(MSVC)
    set(MRAY_CXX_OPTIONS ${MRAY_MSVC_OPTIONS})
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(MRAY_CXX_OPTIONS ${MRAY_CLANG_OPTIONS})
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*GNU")
    set(MRAY_CXX_OPTIONS ${MRAY_GNU_OPTIONS})
endif()

#================#
#  PREPROCESSOR  #
#================#
# Generic Preprocessor Definitions
set(MRAY_PREPROCESSOR_DEFS_GENERIC
    $<$<CONFIG:Debug>:METU_DEBUG>
    $<$<CONFIG:Release>:NDEBUG>)

if(MSVC)
    set(MRAY_PREPROCESSOR_DEFS_GENERIC
        ${MRAY_PREPROCESSOR_DEFS_GENERIC}
        -D_UNICODE
        -DUNICODE
        -DNOMINMAX)
endif()

set(MRAY_PREPROCESSOR_DEFS_CUDA
    METU_CUDA
    # Suppress Dynamic Parallelism cudaDeviceSyncWarning
    -D__CDPRT_SUPPRESS_SYNC_DEPRECATION_WARNING
)

if(MRAY_USE_OPTIX)
    set(MRAY_PREPROCESSOR_DEFS_GENERIC
        ${MRAY_PREPROCESSOR_DEFS_GENERIC} -DMRAY_OPTIX)

    # There is not a easiy way to convert "native" to actual cc number
    # so that the optix can load the actual cc version so we add preporcessor directive
    # so that the module loader select native
    # TODO: Change this to environment variable or something better
    # maybe query the native CC from nvcc ? if that is something.
    if(${CMAKE_CUDA_ARCHITECTURES} STREQUAL "native")
        set(MRAY_PREPROCESSOR_DEFS_GENERIC ${MRAY_PREPROCESSOR_DEFS_GENERIC}
            MRAY_OPTIX_USE_NATIVE_CC)
    endif()
endif()




add_library(mray::meta_compile_opts INTERFACE ALIAS meta_compile_opts)
add_library(mray::cuda_extra_compile_opts INTERFACE ALIAS cuda_extra_compile_opts)

# Compile Options
target_compile_options(meta_compile_opts INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:${PROJECT_WARNINGS_CXX}>
        # C warnings
        $<$<COMPILE_LANGUAGE:C>:${PROJECT_WARNINGS_CXX}>
        # Cuda warnings
        $<$<COMPILE_LANGUAGE:CUDA>:${PROJECT_WARNINGS_CUDA}>
)

# Link Options
target_link_options(meta_compile_opts INTERFACE
                    # C++ Link Opts
                    $<$<COMPILE_LANGUAGE:CXX>:/DEBUG>
                    # CUDA Link Options
                    # All kernels started to show this warning
                    # (when a kernel uses a virtual function) even the
                    # Suppress that warning
                    $<DEVICE_LINK:--nvlink-options=-suppress-stack-size-warning>
                    # Misc shows register usage etc. disabled since it clutters
                    # the compilation and mangled names it hard to follow
                    # Can we generate this as a step maybe?
                    #add_link_options($<DEVICE_LINK:--nvlink-options=--verbose>)
                    #add_link_options($<DEVICE_LINK:--resource-usage>
)

target_compile_definitions(meta_compile_opts
                           INTERFACE
                           ${MRAY_PREPROCESSOR_DEFS_GENERIC}
)

# This defines METU_CUDA
target_compile_definitions(cuda_extra_compile_opts
                           INTERFACE
                           ${MRAY_PREPROCESSOR_DEFS_CUDA}
)

# Meta include dirs
target_include_directories(meta_compile_opts
                           INTERFACE
                           ${MRAY_SOURCE_DIRECTORY}
                           SYSTEM INTERFACE
                           ${MRAY_LIB_INCLUDE_DIRECTORY}
)

# Cmake auto includes this on MSVC but not on Linux?
# Probably cuda installer adds it to path or w/e
if(UNIX NOT APPLE)

    target_include_directories(meta_compile_opts
                                INTERFACE
                                ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

endif()

# Meta Libraries
if(MSVC)
    # Currently no platform specific libraries for windows
    # Besides the cmake finds defualt
    # Probably Windows SDK etc..
elseif(UNIX NOT APPLE)
    # MRay utilizes std::execution c++ features
    # throughout its targets
    # on clang/gcc (at least on linux)
    # this requires tbb to be linked with the system
    # we add this as a meta target on linux
    find_package(tbb::tbb REQUIRED)
    set(MRAY_PLATFORM_SPEC_LIBRARIES
        ${MRAY_PLATFORM_SPEC_LIBRARIES}
        tbb:tbb)
endif()

target_link_libraries(meta_compile_opts INTERFACE
                        ${MRAY_PLATFORM_SPEC_LIBRARIES})

endblock()