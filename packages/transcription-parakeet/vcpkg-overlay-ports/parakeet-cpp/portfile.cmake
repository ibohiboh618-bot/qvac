# parakeet-cpp: NVIDIA Parakeet ASR + Sortformer diarization in pure C++/ggml.
# Sourced from the parakeet-cpp/ subfolder of tetherto/qvac-ext-lib-whisper.cpp;
# consumes the ggml-speech port.
#
# Pinned at 628022ee (DO-NOT-MERGE diagnostic branch off the host-decode commit
# bb585eb1): removes the Mali->CPU guard so ARM Mali (Valhall) runs on Vulkan.
# Round 3h FIXED the encoder miscompute (subsampler depthwise reformulated as an
# elementwise ggml_mul + ggml_sum_rows, no broadcast mul_mat) -- CTC/TDT/EOU now
# correct on Mali and the encoder bisect is clean to encoder_out. The residual is
# the Sortformer diarization head (a SEPARATE ggml graph) outputting "[No speakers
# detected]" on Mali. Round 3i adds a one-shot per-stage GPU-vs-CPU bisect of that
# head (sf_encoder_proj / sf_block0_out / sf_block_last_out / sf_head_h2h /
# sf_logits / sf_probs), both fed the same encoder_out, to name the broken op.
# Diagnostic-only -- NOT for merge. Pairs with ggml-speech 44fd4817 (clean ref).

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO tetherto/qvac-ext-lib-whisper.cpp
    REF 628022ee998e68554b794b6efb0d2c874c2c52e5
    SHA512 41a229386c9cf473b53817f5ae5bdbc3c93a3b8e47c544168b5ad22d20e147469361b5f5e41f88e62417ed472afeeb89d67df0fa637772463b6efaf01e05e589
    HEAD_REF master
)

set(SOURCE_PATH "${WHISPER_CPP_SRC}/parakeet-cpp")
if (NOT EXISTS "${SOURCE_PATH}/CMakeLists.txt")
    message(FATAL_ERROR
        "parakeet-cpp: ${SOURCE_PATH}/CMakeLists.txt missing; the parakeet-cpp/ "
        "subfolder layout in qvac-ext-lib-whisper.cpp may have changed.")
endif()

set(GGML_METAL  OFF)
set(GGML_VULKAN OFF)
set(GGML_CUDA   OFF)
set(GGML_OPENCL OFF)
if("metal" IN_LIST FEATURES)
    set(GGML_METAL ON)
endif()
if("vulkan" IN_LIST FEATURES)
    set(GGML_VULKAN ON)
endif()
if("cuda" IN_LIST FEATURES)
    set(GGML_CUDA ON)
endif()
if("opencl" IN_LIST FEATURES)
    set(GGML_OPENCL ON)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS
        -DPARAKEET_BUILD_LIBRARY=ON
        -DPARAKEET_BUILD_EXECUTABLES=OFF
        -DPARAKEET_BUILD_TESTS=OFF
        -DPARAKEET_BUILD_EXAMPLES=OFF
        -DPARAKEET_INSTALL=ON
        -DPARAKEET_USE_SYSTEM_GGML=ON
        -DBUILD_SHARED_LIBS=OFF
        -DGGML_NATIVE=OFF
        -DGGML_OPENMP=OFF
        -DPARAKEET_OPENMP=OFF
        -DGGML_CCACHE=OFF
        -DPARAKEET_CCACHE=OFF
        -DGGML_METAL=${GGML_METAL}
        -DGGML_VULKAN=${GGML_VULKAN}
        -DGGML_CUDA=${GGML_CUDA}
        -DGGML_OPENCL=${GGML_OPENCL}
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME parakeet-cpp CONFIG_PATH share/parakeet-cpp)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

if (VCPKG_LIBRARY_LINKAGE MATCHES "static")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/bin")
    file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
