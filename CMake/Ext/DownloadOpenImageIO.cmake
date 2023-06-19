
# This one is tedious
# It requires boost :(

# Clone boost
set(OIIO_BOOST_SUBMODULE_LIST

    tools/cmake

    libs/filesystem libs/assert libs/bind
    libs/config libs/container_hash libs/core
    libs/describe libs/mp11 libs/static_assert
    libs/type_traits libs/throw_exception libs/detail
    libs/preprocessor libs/io libs/iterator
    libs/smart_ptr libs/concept_check libs/move
    libs/conversion libs/system libs/predef
    libs/typeof libs/function_types libs/variant2
    libs/atomic libs/mpl libs/fusion
    libs/align libs/tuple libs/optional
    libs/utility libs/functional libs/function
)

# Platform specficic
if(WIN32)

  list(APPEND OIIO_BOOST_SUBMODULE_LIST libs/winapi)

endif()

# Now external project add boost with proper flags
# add a pre-configure custom step that fetches these submodules
# then compile and run

# For OIIO, we embed everything to its dll so we do not need these files to
# be present after the mega lib generation. So override install directory
# of these modules (even do not do install at all)

# And finally call the actual oiio external project add