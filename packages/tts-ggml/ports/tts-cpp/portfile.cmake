# tts-cpp — LOCAL OVERLAY PORT (Chatterbox Mali-GPU verify; DO NOT MERGE).
#
# ROUND 2: pins the ACTUAL Bug-2 fix branch + a verify harness. The PR fix
# (PR #67, branch QVAC-20557-chbx-mali-fix) is master + an is_arm_mali gate that
# auto-routes the CFM attention off the (Mali-miscomputing) f32 flash_attn_ext to
# the unfused soft_max+matmul on Mali only. This overlay pins
# QVAC-20557-chbx-mali-fix-verify = that fix + the S3GEN_DIAG per-stage trace, so
# this device round confirms the GATE auto-fires (no env): expect is_mali=1,
# cfm_unfused=1, f0 bad=0. TTS_CPP_CHBX_CFM_FA=1 forces the broken fused path for
# the A/B control. The S3GEN_DIAG harness is verify-only and is NOT in the fix PR.
# ggml-speech is consumed UNCHANGED from the registry (origin/speech).
#
# Pinned at tetherto/qvac-ext-lib-whisper.cpp branch
# QVAC-20557-chbx-mali-fix-verify (off the PR-#67 fix branch). REF/SHA512 are
# filled by ~/workstuff/overlay-bump.sh after the branch is pushed.
#
# TEMPORARY: this overlay (and its overlay-ports entry in
# vcpkg-configuration.json) is throwaway verify scaffolding — never merge.

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO tetherto/qvac-ext-lib-whisper.cpp
    REF 07e9fc3a0df17cde780aa0305c586d74d18623be
    SHA512 b52e35f9449abc5ed6d5d63cd9f3769876c363ae5badd4e7f20aa2b64034042cf397616d5ec6ed761547726a4fa617aab1129544c85809ae863c9b3ddc7628ba
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
