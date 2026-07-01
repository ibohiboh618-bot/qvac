# tts-cpp — LOCAL-PATH OVERLAY PORT (Mali/Vulkan GPU verify; DO NOT MERGE).
#
# Builds tts-cpp straight from the local worktree
#   /Users/pratiknarola/workstuff/_wt-whisper-mali-verify/tts-cpp
# (branch QVAC-20557-chatterbox-mali-verify, off master a85a4444) so the
# remote-gpu-verify kit's prebuild carries the QVAC-20557 measurement build that
# is NOT (and never to be) published to qvac-registry-vcpkg:
#   - allow_arm_mali=true for Chatterbox T3 (main.cpp) + S3Gen (chatterbox_tts.cpp)
#     to ADMIT Chatterbox onto ARM Mali/Immortalis Vulkan for measurement.
#   - per-stage [gpu-diag] trace (backend_util.h diag_emit): one rms/min/max/nan
#     line per stage straight to the Android log (tag "qvac-tts"), gated by
#     $TTS_CPP_GPU_TRACE, for Supertonic + Chatterbox (T3 + S3Gen) + a VKDEV line.
#   - env-toggled Mali GEMM output-pad (mulmat_pad.h pad_mul_mat, $TTS_CPP_MALI_PAD
#     0=raw / 1=padded) routed through Chatterbox T3 + S3Gen GEMM sites so the kit
#     can run a RAW and a PADDED Mali pass; Supertonic keeps its shipped st_mul_mat.
#
# Pointed at the local checkout (no GitHub fetch / SHA512) for fast local
# iteration. CMake builds out-of-source so the worktree stays clean.
# THROWAWAY measurement scaffolding — never merge; never publish to the registry.

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

set(SOURCE_PATH "/Users/pratiknarola/workstuff/_wt-whisper-mali-verify/tts-cpp")
if (NOT EXISTS "${SOURCE_PATH}/CMakeLists.txt")
    message(FATAL_ERROR
        "tts-cpp local overlay: ${SOURCE_PATH}/CMakeLists.txt missing — is the "
        "_wt-whisper-mali-verify worktree present?")
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
