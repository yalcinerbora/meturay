include(ExtDownloadDeps.cmake)

# External Dependencies
# Comparably Large dependencies would have option to use
# a the system available version

# Get ZLib
mray_build_ext_dependency_git(
        NAME zlib_ext
        URL "https://github.com/madler/zlib.git"
        TAG "04f42ceca40f73e2978b50e93806c2a18c1281fc" # v1.2.13
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
        TAG "a33701196adfad74917046096bf5a2aa0ab0bb50" # v9.1.0
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
        TAG "5d2754306d67d1e654a1a34e1d2e74439a9d53b3" # 16.06.2023 Latest Commit
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
    #TAG "9c54540a1915bbb9fa2f719124d501791c8d69d7" # 16.06.2023 Latest Commit
    TAG "9519a62dd20799c5493c638d1ef5a6f484e5faf1" # v.5.2.5
    BUILD_ARGS
        -DASSIMP_BUILD_ALL_EXPORTERS_BY_DEFAULT=OFF
        -DASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT=ON
        -DASSIMP_BUILD_TESTS=OFF
        -DASSIMP_BUILD_ZLIB=OFF
        -DASSIMP_INJECT_DEBUG_POSTFIX=OFF
        -DASSIMP_INSTALL=ON
        -DBUILD_SHARED_LIBS=ON
        # Non-unix related
        -DASSIMP_LIB_INSTALL_DIR=${MRAY_PLATFORM_NAME}/$<CONFIG>
        -DASSIMP_INCLUDE_INSTALL_DIR=Include
        -DASSIMP_BIN_INSTALL_DIR=${MRAY_PLATFORM_NAME}/$<CONFIG>
        # Windows related
        -DLIBRARY_SUFFIX:STRING=
        -DUSE_STATIC_CRT=OFF
        # There is a bug on this one maybe?
        -DASSIMP_INSTALL_PDB=OFF

        DEPENDENCIES
            zlib_ext
)

# Get GFG
mray_build_ext_dependency_git(
    NAME gfg_ext
    URL "https://github.com/yalcinerbora/GFGFileFormat.git"
    TAG "ae5fd9bee60d70f6a7221c689653a4d00a2bd8cf" # 17.06.2023 Latest Commit
    SOURCE_SUBDIR "Build/CMake"
    BUILD_ARGS
        -DGFG_INSTALL=ON
        -DGFG_BUILD_MAYA_PLUG_IN=OFF
)

# Get spdlog
mray_build_ext_dependency_git(
    NAME spdlog_ext
    URL "https://github.com/gabime/spdlog.git"
    TAG "ad0e89cbfb4d0c1ce4d097e134eb7be67baebb36" # v.1.11.0
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
    URL "https://github.com/OpenImageIO/oiio.git"
    TAG "cfaccc1609266c91aabb03470287a3ada7178cfc" # v.2.5.2.0-dev
    BUILD_ARGS
        -DUSE_PYTHON=OFF
        -DOIIO_BUILD_TESTS=OFF
        -DOIIO_BUILD_TOOLS=OFF
        -DOIIO_DISABLE_BOOST_STACKTRACE=ON
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



