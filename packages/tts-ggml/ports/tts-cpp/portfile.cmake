# tts-cpp — LOCAL OVERLAY PORT (Android GPU CORRECTNESS MEASUREMENT; DO NOT MERGE).
#
# Replaces the registry tts-cpp port so the tts-ggml prebuild builds the
# QVAC-20557 GPU-correctness measurement instrumentation not (and never to be)
# published to qvac-registry-vcpkg. Pins
# tetherto/qvac-ext-lib-whisper.cpp @ b9f9268c (branch
# QVAC-20557-chatterbox-mali-gpu, off PR #54 master b95ad447):
#   - shared detail::diag_stats(): bit-pattern NaN/Inf (fast-math safe) +
#     rms/min/max, emitting one `[gpu-diag] <name> ...` line per stage.
#   - Supertonic + Chatterbox: per-stage [gpu-diag] dumps tagged by backend reg
#     name, emitted on BOTH GPU and CPU runs (gated on $TTS_CPP_GPU_TRACE) so the
#     device-farm logcat can be diffed GPU-vs-CPU to localize the first stage a
#     GPU backend miscomputes.
#   - allow_arm_mali flipped false->true for Chatterbox T3 + S3Gen to ADMIT
#     Chatterbox onto ARM Mali Vulkan FOR MEASUREMENT (not a ship decision):
#     this round MEASURES whether Chatterbox is GPU-correct on Mali rather than
#     assuming it miscomputes. Supertonic already shipped allow_arm_mali=true.
# Paired with test/utils/correlation-helper.js + gpu-smoke.test.js GPU-vs-CPU
# correlation gate. ggml-speech overlay unchanged.
#
# TEMPORARY: this entire overlay (and the overlay-ports entry in
# vcpkg-configuration.json) is throwaway measurement scaffolding — never merge.

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO tetherto/qvac-ext-lib-whisper.cpp
    REF b9f9268cf82467b1b5e98516f114278f64c39023
    SHA512 8503421fab81b3213e6da2528da4f2761343b879b5d9ebd80c8e6466e64ab2d0345de369d26dc5e81662d1444dbe52f808fb927e2f2e4a45c47f0b6d920f3023
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
