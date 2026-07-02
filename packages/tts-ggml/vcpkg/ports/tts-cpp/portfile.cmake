# ─────────────────────────────────────────────────────────────────────────
# TEMPORARY overlay port for tts-cpp.
#
# Points tts-cpp at the `fix/zh-support` branch of the fork
# ishanvohra2/qvac-ext-lib-whisper.cpp instead of the published registry pin,
# so the tts-ggml addon (and CI) can validate the Chatterbox MTL Chinese
# ("zh") enablement end-to-end BEFORE the change is merged upstream and
# republished through qvac-registry-vcpkg.
#
# Branch : fix/zh-support
# Commit : b3642b9a2483e358877c7988dade75b0388eb267
#
# Once the upstream tts-cpp change is merged and the registry port is bumped,
# DELETE this overlay (packages/tts-ggml/vcpkg/ports/tts-cpp) and drop the
# `overlay-ports` entry from packages/tts-ggml/vcpkg-configuration.json so the
# addon resolves tts-cpp from the registry again.
# ─────────────────────────────────────────────────────────────────────────

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO ishanvohra2/qvac-ext-lib-whisper.cpp
    REF b3642b9a2483e358877c7988dade75b0388eb267
    SHA512 faff90f0256c455168dc99102c6089cd8fe5b89c329f16acf8ad19f103c64215a52a6faf2d1700f2ac6589f9c3896010ae8a5f29e9f0af53feb203659495c733
    HEAD_REF fix/zh-support
)

set(SOURCE_PATH "${WHISPER_CPP_SRC}/tts-cpp")
if (NOT EXISTS "${SOURCE_PATH}/CMakeLists.txt")
    message(FATAL_ERROR
        "tts-cpp: ${SOURCE_PATH}/CMakeLists.txt missing; the tts-cpp/ "
        "subfolder layout in qvac-ext-lib-whisper.cpp may have changed.")
endif()

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        metal   GGML_METAL
        vulkan  GGML_VULKAN
        cuda    GGML_CUDA
        opencl  GGML_OPENCL
)

set(PLATFORM_OPTIONS)

if(NOT VCPKG_TARGET_IS_OSX)
    list(APPEND PLATFORM_OPTIONS
        -DGGML_BLAS=OFF
        -DGGML_ACCELERATE=OFF
        -DCMAKE_DISABLE_FIND_PACKAGE_BLAS=ON
    )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS
        -DTTS_CPP_BUILD_LIBRARY=ON
        -DTTS_CPP_BUILD_SHARED=OFF
        -DTTS_CPP_BUILD_EXECUTABLES=OFF
        -DTTS_CPP_BUILD_TESTS=OFF
        -DTTS_CPP_INSTALL=ON
        -DTTS_CPP_USE_SYSTEM_GGML=ON
        -DBUILD_SHARED_LIBS=OFF
        -DGGML_NATIVE=OFF
        -DGGML_OPENMP=OFF
        -DTTS_CPP_OPENMP=OFF
        -DGGML_CCACHE=OFF
        -DTTS_CPP_CCACHE=OFF
        ${FEATURE_OPTIONS}
        ${PLATFORM_OPTIONS}
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME tts-cpp CONFIG_PATH share/tts-cpp)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

if (VCPKG_LIBRARY_LINKAGE MATCHES "static")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/bin")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
