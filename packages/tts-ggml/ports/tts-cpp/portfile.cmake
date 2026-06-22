# tts-cpp — LOCAL OVERLAY PORT (QVAC-20557 Mali GPU CORRECTNESS DIAGNOSTIC; DO NOT MERGE).
#
# Replaces the registry tts-cpp port so the tts-ggml prebuild builds a
# measurement-only variant that is NEVER published to qvac-registry-vcpkg.
# Pins tetherto/qvac-ext-lib-whisper.cpp @ 055b02cf (branch
# QVAC-20557-mali-gpu-diag), which is the SHIPPED tts-cpp a679c7e7
# (registry tts-cpp 2026-06-19, incl. the Supertonic st_mul_mat Mali pad and
# PR #43 chatterbox iOS-memory work) plus ONLY a measurement delta:
#   - public EngineOptions.diag_sink — the addon injects a sink that calls
#     __android_log_print so per-stage [gpu-diag] lines (rms/min/max + bit-pattern
#     NaN/Inf, on BOTH GPU and CPU runs, tagged by backend) reach the device-farm
#     logcat_full.txt (the only native sink that does; stderr is swallowed).
#   - allow_arm_mali flipped false->true for Chatterbox T3 (main.cpp) + S3Gen
#     (chatterbox_tts.cpp) to ADMIT Chatterbox onto ARM Mali Vulkan FOR
#     MEASUREMENT. Supertonic already ships allow_arm_mali=true.
#   - supertonic_vk_mulmat_selftest: in-APK mul_mat accuracy oracle (F32 +
#     F16-weight passes, M/K sweep, vs host gold) at Supertonic GPU load. Round B
#     found the Mali bug is an output-M<48 cliff (3x16 coopmat-tile boundary).
#   - round C (055b02cf): __attribute__((constructor)) setenv GGML_VK_DISABLE_COOPMAT
#     forces the scalar (non-coopmat) Vulkan mul_mat to test if it dodges the cliff;
#     + logs mulmat_needs_pad / coopmat env to __android_log. Vulkan-only (Adreno OpenCL
#     control unaffected). Diagnostic; a Path-A fix would Mali-gate this in ggml-vulkan.
#   - NO token-pinning (dropped vs the prior d8752687 measurement build).
#
# ggml-speech is NOT overlaid: it resolves from the default registry exactly as
# the shipped tts-ggml (#2706) does (baseline 2026-06-15), so the Adreno/Xclipse
# backend path stays byte-identical to ship and this harness cannot regress it.
#
# TEMPORARY: this entire overlay (and the overlay-ports entry in
# vcpkg-configuration.json) is throwaway diagnostic scaffolding — never merge.

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO tetherto/qvac-ext-lib-whisper.cpp
    REF 349189e6e452d213320876ae64abbe3f2d8e30ba
    SHA512 65b7527e9998356ce070d852d46112fd20bb7660a7bce65869a2881387167b9d34ef75a1c874a2da93080f140135d950aab30b7caa1fdf73d5fe7f4a28d29392
    HEAD_REF master
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
