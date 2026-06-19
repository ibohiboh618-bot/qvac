# tts-cpp — LOCAL OVERLAY PORT (Android GPU CORRECTNESS MEASUREMENT; DO NOT MERGE).
#
# Replaces the registry tts-cpp port so the tts-ggml prebuild builds the
# QVAC-20557 GPU-correctness measurement instrumentation not (and never to be)
# published to qvac-registry-vcpkg. Pins
# tetherto/qvac-ext-lib-whisper.cpp @ d8752687 (branch
# QVAC-20557-chatterbox-mali-gpu, off PR #54 master b95ad447):
#   - shared detail::diag_stats() + diag_dump(): bit-pattern NaN/Inf (fast-math
#     safe) + rms/min/max one `[gpu-diag] <name> ...` line per stage, AND a raw
#     f32 dump of each stage to <diag_dump_dir>/<model>_<backend>_<stage>.f32.
#   - PUBLIC EngineOptions.diag_sink + diag_dump_dir (round 2): the addon injects
#     a sink that calls __android_log_print — the ONLY native sink that reaches
#     the device-farm logcat (fprintf(stderr) is swallowed by the bare/RN-host
#     runtime, which is why round 1 emitted ZERO lines). Per-stage dumps power a
#     GPU-vs-CPU Pearson gate; raw buffers are read back in the JS test.
#   - Chatterbox token-pinning (test-only): EngineOptions.test_pinned_tokens +
#     SynthesisResult.speech_tokens bypass the stochastic T3 decode so GPU and
#     CPU S3Gen runs decode IDENTICAL tokens (hard corr gate).
#   - allow_arm_mali=true for Chatterbox T3 + S3Gen to ADMIT Chatterbox onto ARM
#     Mali Vulkan FOR MEASUREMENT (Supertonic already shipped allow_arm_mali=true).
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
    REF d8752687e315a7a2024e6c6750f37d370f35b968
    SHA512 30abc983213f381ebdbeb17ae6121e6df0eaf4337e83d8051ca62c842021d5bdf1b90525989887fe6c9acb2c9303c77aec4aafa49ee496f895e7762f2fff1964
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
