# tts-cpp — LOCAL OVERLAY PORT (Chatterbox Mali-GPU verify; DO NOT MERGE).
#
# Replaces the registry tts-cpp port so the tts-ggml prebuild carries THREE
# changes vs master, all for the Bug-2 Mali-GPU verify (DO NOT MERGE):
#   1. allow_arm_mali=true for Chatterbox T3 (main.cpp) + S3Gen
#      (chatterbox_tts.cpp) — ADMITS Chatterbox onto ARM Mali/Immortalis Vulkan.
#   2. S3GEN_DIAG=1 — per-stage (mu_T/mel/f0/wav) per-block rms/min/max +
#      bit-pattern NaN/Inf trace to localize the token-32 S3Gen collapse.
#   3. S3GEN_FIX=cfm_unfused — swap the CFM flash_attn_ext for soft_max+matmul
#      (the FA-on-Mali fix-swing). Both env-gated; default off = stock behaviour.
# Everything else is current master. ggml-speech is consumed UNCHANGED from the
# registry (origin/speech). NB: in the S3Gen graph only HiFT is scheduler-routed
# and conv_transpose_1d runs on CPU there; the encoder/CFM (the mel producers)
# run on the Mali GPU, which is where Bug 2 lives.
#
# Pinned at tetherto/qvac-ext-lib-whisper.cpp branch
# QVAC-20557-chbx-mali-gpu-verify-0624 (off master). REF/SHA512 are filled by
# ~/workstuff/overlay-bump.sh after the branch is pushed.
#
# TEMPORARY: this overlay (and its overlay-ports entry in
# vcpkg-configuration.json) is throwaway verify scaffolding — never merge.

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO tetherto/qvac-ext-lib-whisper.cpp
    REF 93dcfdadf13b7b92db661e1728986331d15feea4
    SHA512 0a6da017486f78292432ae85aca995313fe4f5606661de63cb2b2e69ad3a57db2b4bcac0374c73a485a4d2b90f1f983dc31a126c6e0d8a6e1b0392412a48e044
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
