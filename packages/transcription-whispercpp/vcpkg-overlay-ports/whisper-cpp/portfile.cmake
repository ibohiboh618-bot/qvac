vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO tetherto/qvac-ext-lib-whisper.cpp
  # QVAC-19213: testing pin to validate the Adreno 740 Vulkan fix
  # (mul_mat_vec subgroup->shmem on Qualcomm, H001-H008) end-to-end through
  # the qvac mobile CI / device farm before merge. This commit = whisper-cpp
  # 1.8.4.3 (f3102199, incl. QVAC-18993 Android dynamic backends) + the
  # latest ggml upstream sync + the Adreno Vulkan fix (PR #30). Revert to
  # REF v${VERSION} once the fix lands on a release tag.
  REF 4273e2710e877c527a52f1efc78bf0b576662208
  SHA512 fc57be0dd6b2725edd78f4617c967c5b83114e0dae3c4151971f7b226afb506f6b32758ece5a74c0f16b737fe756625e94dfec627745f79ec5d29e6e68b68ff9
  HEAD_REF master
)

if (VCPKG_TARGET_IS_ANDROID)
  # NDK only comes with C headers.
  # Make sure C++ header exists, it will be used by ggml tensor library.
  # Need to determine installed vulkan version and download correct headers
  include(${CMAKE_CURRENT_LIST_DIR}/android-vulkan-version.cmake)
  detect_ndk_vulkan_version()
  message(STATUS "Using Vulkan C++ wrappers from version: ${vulkan_version}")
  file(DOWNLOAD
    "https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/v${vulkan_version}.tar.gz"
    "${SOURCE_PATH}/vulkan-sdk-${vulkan_version}.tar.gz"
    TLS_VERIFY ON
  )

  file(ARCHIVE_EXTRACT
    INPUT "${SOURCE_PATH}/vulkan-sdk-${vulkan_version}.tar.gz"
    DESTINATION "${SOURCE_PATH}"
  )

  # Copy the Vulkan headers to where the build system expects them
  # The build system looks for vulkan/vulkan.hpp with include path pointing to ggml/src/
  file(COPY "${SOURCE_PATH}/Vulkan-Headers-${vulkan_version}/include/"
       DESTINATION "${SOURCE_PATH}/ggml/src/")
  
  # Clean up the temporary extracted directory
  file(REMOVE_RECURSE "${SOURCE_PATH}/Vulkan-Headers-${vulkan_version}")
endif()

set(PLATFORM_OPTIONS)

if (VCPKG_TARGET_IS_OSX)
  list(APPEND PLATFORM_OPTIONS -DGGML_METAL=ON)
elseif (VCPKG_TARGET_IS_IOS)
  # Intentionally NOT -DGGML_METAL=ON. iOS bare-kit builds were hitting
  # a separate Metal/Compiler XPC crash during transcribe() on physical
  # iPhone (XPC_ERROR_CONNECTION_INTERRUPTED / MTLCompiler peer-unloaded)
  # that is being investigated independently of the OutputCallBackJs
  # teardown UAF. Force the flag OFF so it overrides any upstream default
  # and stays explicit in the build log; iOS falls back to the CPU
  # backend until the Metal-side issue is fixed.
  list(APPEND PLATFORM_OPTIONS -DGGML_METAL=OFF)
elseif("vulkan" IN_LIST FEATURES)
  list(APPEND PLATFORM_OPTIONS -DGGML_VULKAN=ON)
else()
  list(APPEND PLATFORM_OPTIONS -DGGML_VULKAN=OFF)
endif()

# Android: ship the same dynamic-backend + CPU-variant recipe llama-cpp
# already uses on this triplet. GGML_BACKEND_DL=ON makes ggml load the
# backend implementations as separate .so files at runtime (one per
# backend, picked by the device caps), so a single APK ships all the
# variants and the consumer's binary only statically links the dispatcher.
# GGML_CPU_ALL_VARIANTS + GGML_CPU_REPACK gives one tuned CPU .so per
# microarch (armv8.0/armv8.2-fp16/armv8.2-fp16+dotprod/armv8.7-i8mm), and
# COOPMAT[2] are disabled because the Vulkan validation layer's coopmat
# extensions are unstable on Adreno NDK headers.
# OpenCL is gated behind the `opencl` feature so non-Adreno Android
# consumers don't pull in an unused backend.
# Android dynamic-backend mode: per-microarch CPU + GPU backends ship as
# MODULE .so files dlopen'd at runtime, while the dispatcher
# (libwhisper.a, libggml.a, libggml-base.a) stays static — same shape
# as the speech-stack uses for parakeet-cpp/tts-cpp.
#
# The REF pin (QVAC-19213, PR #30) is whisper-cpp 1.8.4.3 with QVAC-18993
# already merged in, so the GGML_BACKEND_DL combo above works end-to-end on
# Android. The two QVAC-18993 commits (now on tetherto/master as part of
# 1.8.4.3) are:
#   400bf929  ggml : allow GGML_BACKEND_DL with a static core
#             (removes the FATAL_ERROR + flips PIC/GGML_BUILD on)
#   c1d7a6c9  ggml-backend : android per-arch CPU variant dlopen fallback
#             (lets ggml_backend_load_best resolve libggml-cpu-android_armv*_*.so
#              via Android's in-APK linker when there's no on-disk lib dir)
# The QVAC-19213 REF merges tetherto/master (1.8.4.3) in on top of the ggml
# upstream sync; re-point to REF v${VERSION} once the Adreno fix is tagged.
if(VCPKG_TARGET_IS_ANDROID)
  set(DL_BACKENDS ON)
  list(APPEND PLATFORM_OPTIONS
    -DGGML_BACKEND_DL=ON
    -DGGML_CPU_ALL_VARIANTS=ON
    -DGGML_CPU_REPACK=ON
    -DGGML_VULKAN_DISABLE_COOPMAT=ON
    -DGGML_VULKAN_DISABLE_COOPMAT2=ON)
  if("opencl" IN_LIST FEATURES)
    list(APPEND PLATFORM_OPTIONS -DGGML_OPENCL=ON)
  endif()
else()
  set(DL_BACKENDS OFF)
endif()

# spirv-headers include shim: ggml-vulkan.cpp #includes
# <spirv/unified1/spirv.hpp> unconditionally, but ggml-vulkan's
# CMakeLists.txt does not call find_package(SpirvHeaders), so the
# vcpkg-installed spirv-headers include prefix must be added explicitly.
#
# QVAC-19213: add it via CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES (a dedicated
# system-include variable, emitted as -isystem / /external:I per compiler for
# every C++ target) instead of stuffing it into CMAKE_CXX_FLAGS. Passing
# `-DCMAKE_CXX_FLAGS=...` *replaced* the triplet's VCPKG_CXX_FLAGS — notably
# `-stdlib=libc++` on the x64-linux / arm64-linux triplets — so ggml/whisper
# built against GNU libstdc++ while the addon links libc++, producing
# undefined-symbol link errors (std::__cxx11::*, std::filesystem::__cxx11::*,
# std::_V2::system_category) in the addon's C++ unit tests. This variable
# leaves CMAKE_CXX_FLAGS untouched, so the triplet's stdlib choice is honored
# on every platform (Android keeps using the NDK libc++ as before).
set(SPIRV_HEADERS_CFLAGS "")
if("vulkan" IN_LIST FEATURES)
  set(SPIRV_HEADERS_CFLAGS "-DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES=${CURRENT_INSTALLED_DIR}/include")
endif()

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  DISABLE_PARALLEL_CONFIGURE
  OPTIONS
    -DGGML_CCACHE=OFF
    -DGGML_OPENMP=OFF
    -DGGML_NATIVE=OFF
    -DWHISPER_BUILD_TESTS=OFF
    -DWHISPER_BUILD_EXAMPLES=OFF
    -DWHISPER_BUILD_SERVER=OFF
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_BUILD_NUMBER=1
    ${PLATFORM_OPTIONS}
    ${SPIRV_HEADERS_CFLAGS}
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(
  PACKAGE_NAME whisper
  CONFIG_PATH share/whisper
)

vcpkg_fixup_pkgconfig()

vcpkg_copy_pdbs()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

if (NOT DL_BACKENDS AND VCPKG_LIBRARY_LINKAGE MATCHES "static")
  # On dynamic-backend Android the ggml backend .so files live in bin/
  # alongside the static dispatcher; wiping bin/ here would silently
  # ship a runtime that loads no backends. Only wipe for true
  # static-only triplets.
  file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/bin")
  file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/bin")
endif()

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")