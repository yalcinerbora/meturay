


# For OIIO, we embed everything to its dll so we do not need these files to
# be present after the mega lib generation. So override install directory
# of these modules (even do not do install at all)

# And finally call the actual oiio external project add

# Install only OIIO related libraries to a different folder
# this folder will be deleted after everything is installed
# Thus, all of these libraries should be static library
set(OIIO_LIB_DIR ${MRAY_PLATFORM_EXT_DIRECTORY}/tmp/lib)

# Append to prefix path here for lib finding
set(OIIO_LIB_CONFIG_DIR ${OIIO_LIB_DIR}/${MRAY_PLATFORM_NAME}/$<CONFIG>)
append_cmake_prefix_path(${OIIO_LIB_CONFIG_DIR}/cmake)
append_cmake_prefix_path(${OIIO_LIB_CONFIG_DIR}/pkgconfig)

# This one is tedious. It requires boost :(
# Now external project add boost with proper flags and submodules
# Clone boost
# Come on boost, too many deps :(
set(OIIO_BOOST_SUBMODULE_LIST
    # CMake
    tools/cmake
    # FS Related
    libs/filesystem
    libs/assert libs/bind libs/function
    libs/config libs/container_hash libs/core
    libs/describe libs/mp11 libs/static_assert
    libs/type_traits libs/throw_exception libs/detail
    libs/preprocessor libs/io libs/iterator
    libs/smart_ptr libs/concept_check libs/move
    libs/conversion libs/system libs/predef
    libs/typeof libs/function_types libs/variant2
    libs/atomic libs/mpl libs/fusion
    libs/align libs/tuple libs/optional
    libs/utility libs/functional
    # Thread Related
    libs/thread
    libs/chrono libs/integer libs/container
    libs/ratio libs/intrusive libs/date_time
    libs/algorithm libs/rational libs/exception
    libs/array libs/lexical_cast libs/regex
    libs/range libs/unordered libs/numeric/conversion
    libs/tokenizer

    #
    libs/headers
)
# Platform specific
if(WIN32)
  list(APPEND OIIO_BOOST_SUBMODULE_LIST libs/winapi)
endif()

mray_build_ext_dependency_git(
        NAME boost_fs_thrd_ext
        URL "https://github.com/boostorg/boost.git"
        TAG "b6928ae5c92e21a04bbe17a558e6e066dbe632f6" # v1.82.0
        # We will only download some of the boost
        SPECIFIC_SUBMODULES ${OIIO_BOOST_SUBMODULE_LIST}

        LICENSE_NAME "LICENSE_1_0.txt"
        OVERRIDE_INSTALL_PREFIX ${OIIO_LIB_DIR}
        BUILD_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -DBOOST_INCLUDE_LIBRARIES:STRING=filesystem|thread
)

# OpenEXR
mray_build_ext_dependency_git(
        NAME imath_ext
        URL "https://github.com/AcademySoftwareFoundation/Imath.git"
        TAG "d690a3fcff4e877ead5ae56c7e964595ade8a35e" # v3.1.9

        LICENSE_NAME "LICENSE.md"
        OVERRIDE_INSTALL_PREFIX ${OIIO_LIB_DIR}
        BUILD_ARGS
            -DBUILD_SHARED_LIBS=OFF
            #-DCMAKE_DEBUG_POSTFIX:STRING=
)

mray_build_ext_dependency_git(
        NAME openexr_ext
        URL "https://github.com/AcademySoftwareFoundation/openexr.git"
        TAG "68d9e1e17620cef00e59b43fa42c97fbcf90e72b" # v3.1.8

        LICENSE_NAME "LICENSE.md"
        OVERRIDE_INSTALL_PREFIX ${OIIO_LIB_DIR}
        BUILD_ARGS
            -DBUILD_SHARED_LIBS=OFF
            #-DCMAKE_DEBUG_POSTFIX:STRING=
            -DOPENEXR_BUILD_TOOLS=OFF
            -DOPENEXR_FORCE_INTERNAL_IMATH=OFF
            -DOPENEXR_INSTALL=ON
            -DOPENEXR_INSTALL_EXAMPLES=OFF
            -DOPENEXR_INSTALL_PKG_CONFIG=OFF
            -DOPENEXR_INSTALL_TOOLS=OFF
            -DZLIB_USE_STATIC_LIBS=ON

        DEPENDENCIES
            imath_ext
            zlib_ext
)

# Lib TIFF
mray_build_ext_dependency_git(
        NAME libtiff_ext
        URL "https://gitlab.com/libtiff/libtiff.git"
        TAG "38eb7b00cb5767770017fb91743a960ffd96d774" # v4.5.0

        LICENSE_NAME "LICENSE.md"
        OVERRIDE_INSTALL_PREFIX ${OIIO_LIB_DIR}
        BUILD_ARGS
            -DBUILD_SHARED_LIBS=OFF
            -Dtiff-tools=OFF
            -Dtiff-tests=OFF
            -Dtiff-docs=OFF

)

# Lib JPEG-turbo
mray_build_ext_dependency_git(
        NAME libjpeg_turbo_ext
        URL "https://github.com/libjpeg-turbo/libjpeg-turbo.git"
        TAG "8ecba3647edb6dd940463fedf38ca33a8e2a73d1" # v2.1.5.1

        LICENSE_NAME "LICENSE.md"
        OVERRIDE_INSTALL_PREFIX ${OIIO_LIB_DIR}
        BUILD_ARGS
            -DENABLE_SHARED=OFF
)

# Lib PNG
# TODO: Add Jpeg2000, HEIF, HDR

# And finally oiio
mray_build_ext_dependency_git(
        NAME oiio_ext
        URL "https://github.com/OpenImageIO/oiio.git"
        TAG "e7fe4064814e1758e47cb4700e33ef930673a427" # v2.4.12.0

        LICENSE_NAME "LICENSE.md"
        BUILD_ARGS
            -DBoost_USE_STATIC_LIBS=ON
            -DBoost_NO_WARN_NEW_VERSIONS=ON
            -DEMBEDPLUGINS=ON
            -DUSE_PYTHON=OFF
            -DUSE_OPENGL=ON
            -DUSE_QT=OFF
            -DBUILD_SHARED_LIBS=ON
            -DLINKSTATIC=ON
            -DBUILD_MISSING_FMT=OFF # We have a fmt
            -DALWAYS_PREFER_CONFIG=ON
            -DOIIO_BUILD_TESTS=OFF
            -DOIIO_BUILD_TOOLS=OFF
            -DBUILD_DOCS=OFF
            -DBUILD_TESTING=OFF
            -DINSTALL_DOCS=OFF
            #-DOIIO_LOCAL_DEPS_PATH=${CMAKE_PREFIX_PATH}
            -DOIIO_DISABLE_BOOST_STACKTRACE=ON

            -DImath_ROOT:PATH=${OIIO_LIB_CONFIG_DIR}/cmake/Imath

            -DVERBOSE=ON

        DEPENDENCIES
            boost_fs_thrd_ext
            zlib_ext
            openxr_ext
            libtiff_ext
            libjpeg_turbo_ext
            fmt_ext
)

# Issue of https://github.com/OpenImageIO/oiio/pull/3862
# It is not merged but we can patch it!
# I will add this as an extra step since we cant modify
# patch step after.
set(MRAY_OIIO_SRC_LOCATION ${MRAY_PLATFORM_EXT_DIRECTORY}/oiio_ext/src/oiio_ext)

# DL'in file here instead
file(DOWNLOAD https://github.com/OpenImageIO/oiio/pull/3862.patch
              ${MRAY_OIIO_SRC_LOCATION}/../3862.patch)

ExternalProject_Add_Step(oiio_ext patch_extra
                WORKING_DIRECTORY ${MRAY_OIIO_SRC_LOCATION}/
                # Couldnt able to make this work
                # COMMAND curl --create-dirs -s -o ${MRAY_OIIO_SRC_LOCATION}/3862.patch
                #         "https://github.com/OpenImageIO/oiio/pull/3862.patch"
                # Download patch at configure step, then copy here
                #
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${MRAY_OIIO_SRC_LOCATION}/../3862.patch
                        .
                # This is also not good, it will move head back and forth but this tecnhically
                # will run only initially so it is kinda fine.
                COMMAND git am 3862.patch
                DEPENDEES DOWNLOAD UPDATE
                DEPENDERS PATCH CONFIGURE
                COMMENT "Applying a patch to oiio"
)

