# tts-cpp — LOCAL OVERLAY PORT (Android GPU validation; DO NOT MERGE).
#
# Replaces the registry tts-cpp port so the tts-ggml prebuild builds the
# Android-GPU work not yet published to qvac-registry-vcpkg. Pins
# tetherto/qvac-ext-lib-whisper.cpp @ 45602048 (PR #54 — the consolidated
# QVAC-20557 Android-GPU stack on master):
#   - dlopen reroute: route Supertonic's direct CPU-backend calls
#     (from_float / backend_is_cpu) through ggml-base under GGML_BACKEND_DL=ON
#     (else the addon SIGABRTs at dlopen on Android).
#   - keep Supertonic K/V attention/weights F32 on OpenCL (no F32xF16 mat-vec).
#   - explicit GPU attention when the backend can't run flash-attn (Adreno /
#     Xclipse route FLASH_ATTN_EXT to CPU) so CFM attention stays GPU-resident.
#   - allowlist Samsung Xclipse (Vulkan) + a gpu_unsupported() policy-decline.
#   - ARM Mali / Valhall: model-side st_mul_mat output-pad works around the
#     driver's small-output-dim mul_mat miscompute; Mali keeps weights F32 (the
#     pad only covers F32 operands); Mali-Vulkan is allowlisted for Supertonic
#     only (Chatterbox stays CPU on Mali via gpu_unsupported).
# ggml-speech overlay stays stock at 44fd4817 — no ggml change.
#
# TEMPORARY: remove this overlay (and the overlay-ports entry in
# vcpkg-configuration.json) once the fix is published to the registry and
# consumed via vcpkg.json.

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO tetherto/qvac-ext-lib-whisper.cpp
    REF 4560204843e5d901ca27a6ef0881f6469f917cb1
    SHA512 90326458cb9bd2ab282e8c690a817e60318c04328727904be4d0787ba6159b039566f5954d99a273ee069932941389bef1189fee2f0b7eff627e6de8b9c11cf0
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
