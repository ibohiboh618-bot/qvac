# tts-cpp — LOCAL OVERLAY PORT (Chatterbox Mali-GPU verify; DO NOT MERGE).
#
# Replaces the registry tts-cpp port so the tts-ggml prebuild carries ONE change
# vs master: allow_arm_mali=true for Chatterbox T3 (main.cpp) + S3Gen
# (chatterbox_tts.cpp), to ADMIT Chatterbox onto ARM Mali/Immortalis Vulkan.
# Everything else is current master. ggml-speech is consumed UNCHANGED from the
# registry — origin/speech has no conv_transpose Mali gate, so conv_transpose_1d
# runs on the Mali GPU in this build.
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
    REF bc0847ad59172a2b05198fcfaf56dcfe32ffcc10
    SHA512 0038bf407190752ddfc4df59c99e6d839665fd159b85230d4c9de776119ee17c561087e5cbd27ba3d5e4a7f9a3e113f11ce84742cf39b17cbdf8e261b5e6abe2
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
