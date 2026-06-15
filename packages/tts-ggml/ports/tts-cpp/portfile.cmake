# tts-cpp — LOCAL OVERLAY PORT (MALI-VULKAN PROBE; DO NOT MERGE).
#
# THROWAWAY probe branch: pins tetherto/qvac-ext-lib-whisper.cpp @ 0aa594a6 =
# the QVAC-20557 stack (below) + one extra commit that allowlists ARM Mali so
# the device farm (Pixel 9 / Mali) attempts Mali Vulkan and surfaces its failure
# mode (abort vs garbage) in the logs. The real PR (#2605) pins aa2c9056 (no
# Mali). Builds the Android-GPU fixes not yet published to qvac-registry-vcpkg:
#   1. dlopen reroute: Supertonic's direct CPU-backend calls are unlinkable
#      under GGML_BACKEND_DL=ON; route ggml_get_type_traits_cpu(...)->from_float
#      to ggml_quantize_chunk() and ggml_backend_is_cpu() to the registry shim
#      tts_cpp::detail::backend_is_cpu() (else the addon SIGABRTs at dlopen on
#      Android, killing all Android e2e).
#   2. keep Supertonic K/V attention F32 on OpenCL (no F32xF16 mat-vec kernel).
#   3. explicit GPU attention when the backend can't run flash-attn (Adreno /
#      Xclipse OpenCL route FLASH_ATTN_EXT to CPU) so the per-step CFM attention
#      stays GPU-resident instead of going stale on a CPU bridge.
#   4. allowlist Samsung Xclipse (Vulkan) as a 2nd Android GPU vendor + report a
#      gpu_unsupported() policy-decline fallback (Mali stays CPU).
#
# TEMPORARY: remove this overlay (and the overlay-ports entry in
# vcpkg-configuration.json) once the fixes are published to the registry and
# consumed via vcpkg.json.

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO tetherto/qvac-ext-lib-whisper.cpp
    REF 0aa594a6fa18bd61271c07197fbe2178f750f3d2
    SHA512 9df14ca4b0174ccfc3aa99c62041f1a34c2be6bbf03c4ee4fc6778745455aadb9142152276b36cb9e1c8f2fc1ed511c22ac35baaf10da966424cee356f75abdf
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
