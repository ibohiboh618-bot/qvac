# parakeet-cpp: NVIDIA Parakeet ASR + Sortformer diarization in pure C++/ggml.
# Sourced from the parakeet-cpp/ subfolder of tetherto/qvac-ext-lib-whisper.cpp;
# consumes the ggml-speech port.
#
# Pinned at 8146f1bc (DO-NOT-MERGE diagnostic branch off the host-decode commit
# bb585eb1): removes the Mali->CPU guard so ARM Mali (Valhall) runs on Vulkan.
# Round 3h FIXED the encoder miscompute (subsampler depthwise reformulated as an
# elementwise ggml_mul + ggml_sum_rows, no broadcast mul_mat) -- CTC/TDT/EOU now
# correct on Mali. Round 3i localised the Sortformer residual to transformer
# block 0 (sf_block0_out -> NaN on Mali; sf_encoder_proj clean; Adreno/Metal
# clean). Round 3j routed the diarization head to the CPU backend but its
# GPU-resident q4_0 weights crashed the CPU matmul. Round 3k (Option B) builds
# CPU-resident copies of the head weights at load on Mali-Vulkan and reads them
# when the head runs on CPU; the encoder + CTC/TDT/EOU stay on the Mali GPU. The
# encoder bisect + diag logs stay for on-device validation.
# Diagnostic-only -- NOT for merge. Pairs with ggml-speech 44fd4817 (clean ref).

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
set(VCPKG_BUILD_TYPE release)

vcpkg_from_github(
    OUT_SOURCE_PATH WHISPER_CPP_SRC
    REPO tetherto/qvac-ext-lib-whisper.cpp
    REF 8146f1bcb5bfe59c615b96e2fe7643aa62c859cc
    SHA512 8108c30e1dabb93639c803fa572e9b0273e8ac9285da06c27cf7d49d751b3fee5ba82a7dad527b88e1529a68140df970844e2a8335f869bf7d4a5017d9c6892a
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
