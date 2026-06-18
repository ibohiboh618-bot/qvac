# tts-cpp — LOCAL OVERLAY PORT (0.2.x EOS backport, QVAC-21056).
#
# Pins qvac-ext-lib-whisper.cpp@f7d4d6c (the 0.2.4 base) and applies the
# QVAC-20616 end-of-speech fix on top as
# `patches/0001-qvac-20616-eos-stop.patch`.
#
# QVAC-21056 backports QVAC-20616 ("random tokens appearing after inference")
# to the 0.2.x line. The patch is the exact f7d4d6c..ea51e37b diff: Phase 1
# heuristic stop controller (EOS confidence + n-gram repetition + text-length
# budget), Phase 2 alignment-based EOS (ports the reference
# AlignmentStreamAnalyzer cross-attention signal via an in-graph attention
# probe), improvements #1-#5 and their unit/round-trip tests. It is the same
# fix that landed upstream in tts-cpp master b95ad447 (PR #53) and that the
# 0.3.x line consumes from the registry as `tts-cpp 2026-06-18`; here it is
# carried as an in-package patch so 0.2.5 stays on the proven f7d4d6c base
# (no registry/master bump). Stops the Chatterbox multilingual model rambling
# ~20s of random tokens after the intended text.
#
# Base f7d4d6c (inherited from 0.2.4) is the published 2026-06-05 pin 128dae42
# plus the Android dlopen fix: it reroutes Supertonic's direct CPU-backend
# calls that are unlinkable under `GGML_BACKEND_DL=ON` --
#   - `ggml_get_type_traits_cpu(...)->from_float` -> `ggml_quantize_chunk()`
#     (ggml-base, always linked)
#   - `ggml_backend_is_cpu()` -> `tts_cpp::detail::backend_is_cpu()` registry
#     shim (the pattern Chatterbox / parakeet already use)
# so the addon `dlopen`s cleanly on Android (the 0.2.1 bootstrap crash).
#
# How to sanity-check the EOS patch applied: after configure, the build tree's
# tts-cpp sources contain src/t3_alignment_analyzer.* and src/t3_stop_controller.*.
#
# TEMPORARY: this overlay (and the `overlay-ports` entry in
# vcpkg-configuration.json) exist only for the 0.2.x line. master / 0.3.x
# consume the same fix from the registry (tts-cpp 2026-06-18); drop the overlay
# if the 0.2.x line is ever rebased onto a registry pin that carries both fixes.

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO tetherto/qvac-ext-lib-whisper.cpp
    REF f7d4d6c18615dc0c094776db78421bbb07e90371
    SHA512 eb4d5679db948a496282f3a73ad11da0b19efbfc63c0b27a08cb536a88c3313ad03f91aa2f875463fd9990b2cf0e2b66a063a3a84bb2ff930718926c497b435d
    HEAD_REF master
    PATCHES
        patches/0001-qvac-20616-eos-stop.patch
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
