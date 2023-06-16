include(ExtDownloadDeps.cmake)

# External Dependencies
# Comparably Large dependencies would have option to use
# a the system available version

# Get ZLib
mray_build_ext_dependency_git(
        NAME zlib_ext
        URL "https://github.com/madler/zlib.git"
        TAG "04f42ceca40f73e2978b50e93806c2a18c1281fc" #v1.2.13
        BUILD_ARGS
            -DINSTALL_BIN_DIR=${MRAY_CONFIG_LIB_DIRECTORY}
            -DINSTALL_INC_DIR=${MRAY_LIB_INCLUDE_DIRECTORY}
            -DINSTALL_LIB_DIR=${MRAY_CONFIG_LIB_DIRECTORY}
            -DINSTALL_MAN_DIR=${MRAY_LIB_DIRECTORY}/Docs/zlib
            -DINSTALL_PKGCONFIG_DIR=${MRAY_CONFIG_LIB_DIRECTORY}/pkgconfig
)

# Get FMT
mray_build_ext_dependency_git(
        NAME fmt_ext
        URL "https://github.com/fmtlib/fmt.git"
        TAG "a0b8a92e3d1532361c2f7feb63babc5c18d00ef2" #v10.0.0
        BUILD_ARGS
            -DFMT_DEBUG_POSTFIX:STRING=
            -DFMT_DOC=OFF
            -DFMT_TEST=OFF
            # FMT has specific install variables for various stuff
            # set them specifically
            -DFMT_INC_DIR=Include
            -DFMT_LIB_DIR=${MRAY_PLATFORM_NAME}/$<CONFIG>
            -DFMT_CMAKE_DIR=${MRAY_PLATFORM_NAME}/$<CONFIG>/cmake/fmt
            -DFMT_PKGCONFIG_DIR=${MRAY_PLATFORM_NAME}/$<CONFIG>/pkgconfig
)

# Get nlohmann-json
mray_build_ext_dependency_git(
        NAME json_ext
        URL "https://github.com/nlohmann/json.git"
        TAG "5d2754306d67d1e654a1a34e1d2e74439a9d53b3" #16.06.2023 Latest Commit
        BUILD_ARGS
            -DJSON_BuildTests=OFF
            -DBUILD_TESTING=OFF
            -DNLOHMANN_JSON_CONFIG_INSTALL_DIR=${MRAY_CONFIG_LIB_DIRECTORY}/cmake/json
            -DNLOHMANN_JSON_INCLUDE_INSTALL_DIR=${MRAY_LIB_INCLUDE_DIRECTORY}
            -DNLOHMANN_JSON_PKGCONFIG_INSTALL_DIR=${MRAY_CONFIG_LIB_DIRECTORY}/pkgconfig
)

# Get assimp
mray_build_ext_dependency_git(
    NAME assimp_ext
    URL "https://github.com/assimp/assimp.git"
    TAG "9519a62dd20799c5493c638d1ef5a6f484e5faf1" #v.5.2.5
    BUILD_ARGS
        -DASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT=OFF
        -DASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT=ON
        -DASSIMP_BUILD_TESTS=OFF
        -DASSIMP_BUILD_ZLIB=OFF
        -DASSIMP_INJECT_DEBUG_POSTFIX=OFF
        -DASSIMP_INSTALL=ON
        -DBUILD_SHARED_LIBS=ON
        DEPENDENCIES
            zlib_ext
)

# Get spdlog
mray_build_ext_dependency_git(
    NAME spdlog_ext
    URL "https://github.com/gabime/spdlog.git"
    TAG "ad0e89cbfb4d0c1ce4d097e134eb7be67baebb36" #v.1.11.0
    BUILD_ARGS
        -DSPDLOG_BUILD_EXAMPLE=OFF
        -DSPDLOG_BUILD_PIC=ON
        -DSPDLOG_BUILD_SHARED=ON
        # Our fmt is too up to date
        -DSPDLOG_FMT_EXTERNAL=OFF
)

# Image Libraries
mray_build_ext_dependency_git(
    NAME oiio_ext
    URL "https://github.com/gabime/spdlog.git"
    TAG "e7fe4064814e1758e47cb4700e33ef930673a427" #v.2.4.12.0
    BUILD_ARGS
        -DUSE_PYTHON=OFF
        -DOIIO_BUILD_TESTS=OFF
        -DOIIO_BUILD_TOOLS=OFF
        -DOIIO_DISABLE_BOOST_STACKTRACE=ON
        -DSPDLOG_BUILD_PIC=ON
        -DSPDLOG_BUILD_SHARED=ON
        -DEMBED_PLUGINS=ON
        -DCMAKE_DEBUG_POSTFIX:STRING=
        -DENABLE_FAST_MATH=ON
    DEPENDENCIES
            zlib_ext
            boost_ext
            openexr_ext
            libtiff_ext
            libjpeg_ext
            fmt_ext

)



