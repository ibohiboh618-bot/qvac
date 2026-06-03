# ggml-speech overlay -> tetherto/qvac-ext-ggml PR #19 head
# (QVAC-19213-parakeet-opencl @ 89bbeeaa). Builds ggml-speech straight from the
# PR branch so the Adreno OpenCL fix can be exercised pre-merge. Mirrors the
# registry port's per-platform backends (Metal on Apple, Vulkan on desktop,
# OpenCL on Android) so CI matches baseline; only Android is OpenCL-only
# (no Vulkan). Drop this overlay once PR #19 merges and the registry ships it.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO tetherto/qvac-ext-ggml
    REF 89bbeeaa5904a8f30a63c274614b0d6b48dca948
    SHA512 11ec3cb7cf3f15203bfdaa7a4c7db58ccdc3b6f19b6da9670f02071953d1541c225b37dc2ef89b22fbe211a921a627d95631efbd345852d6f28e397b250b6633
    HEAD_REF QVAC-19213-parakeet-opencl
    PATCHES
        # PR #19's ggml-vulkan.cpp resolves <spirv/unified1/spirv.hpp> via
        # __has_include, but its CMakeLists never wires spirv-headers. Carry the
        # registry's fix so the desktop Vulkan build links the vcpkg
        # spirv-headers include dir. Inert on Metal/OpenCL builds (ggml-vulkan
        # is not configured there).
        patches/0001-ggml-vulkan-find-spirv-headers.patch
)

# Per-feature backend selection (mirrors the registry ggml-speech port).
set(GGML_METAL  OFF)
set(GGML_VULKAN OFF)
set(GGML_OPENCL OFF)
set(GGML_METAL_FUSE_MV_BIAS OFF)  # baseline default; fusion zeroes parakeet EOU on Metal
if("metal" IN_LIST FEATURES)
    set(GGML_METAL ON)
endif()
if("vulkan" IN_LIST FEATURES)
    set(GGML_VULKAN ON)
endif()
if("opencl" IN_LIST FEATURES)
    set(GGML_OPENCL ON)
endif()

set(PLATFORM_OPTIONS)
if(VCPKG_TARGET_IS_IOS)
    list(APPEND PLATFORM_OPTIONS -DGGML_BLAS=OFF -DGGML_ACCELERATE=OFF)
endif()
if(VCPKG_TARGET_IS_ANDROID)
    # Hybrid Android backend mode: GPU/CPU backends as MODULE .so loaded at
    # runtime via dlopen (per-arch CPU variants picked at first use).
    list(APPEND PLATFORM_OPTIONS
        -DGGML_BACKEND_DL=ON
        -DGGML_CPU_ALL_VARIANTS=ON
        -DGGML_CPU_REPACK=ON
    )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_SHARED_LIBS=OFF
        -DGGML_NATIVE=OFF
        -DGGML_CCACHE=OFF
        -DGGML_OPENMP=OFF
        -DGGML_LLAMAFILE=OFF
        -DGGML_BUILD_TESTS=OFF
        -DGGML_BUILD_EXAMPLES=OFF
        -DGGML_METAL=${GGML_METAL}
        -DGGML_VULKAN=${GGML_VULKAN}
        -DGGML_CUDA=OFF
        -DGGML_OPENCL=${GGML_OPENCL}
        -DGGML_METAL_FUSE_MV_BIAS=${GGML_METAL_FUSE_MV_BIAS}
        -DGGML_LIB_OUTPUT_PREFIX=qvac-speech-
        ${PLATFORM_OPTIONS}
)

vcpkg_cmake_install()

# Android dynamic-backend mode builds the MODULE backend .so files into the
# buildtree bin/; cmake install() doesn't move them by default.
if(VCPKG_TARGET_IS_ANDROID)
    file(GLOB _backend_sos
        "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/bin/libqvac-speech-ggml-*.so"
    )
    if(_backend_sos)
        file(INSTALL ${_backend_sos} DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
    endif()
endif()

vcpkg_cmake_config_fixup(PACKAGE_NAME ggml CONFIG_PATH lib/cmake/ggml)

if(EXISTS "${CURRENT_PACKAGES_DIR}/share/pkgconfig/ggml.pc")
    file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/lib/pkgconfig")
    file(RENAME "${CURRENT_PACKAGES_DIR}/share/pkgconfig/ggml.pc"
                "${CURRENT_PACKAGES_DIR}/lib/pkgconfig/ggml.pc")
endif()
if(EXISTS "${CURRENT_PACKAGES_DIR}/debug/share/pkgconfig/ggml.pc")
    file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig")
    file(RENAME "${CURRENT_PACKAGES_DIR}/debug/share/pkgconfig/ggml.pc"
                "${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig/ggml.pc")
endif()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/share/pkgconfig"
                    "${CURRENT_PACKAGES_DIR}/debug/share/pkgconfig")
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
