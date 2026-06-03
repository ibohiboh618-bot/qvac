set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)

set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/../toolchains/linux-clang.cmake")

set(_qvac_c_flags "-fPIC")
set(_qvac_cxx_flags "-fPIC -stdlib=libc++")
set(_qvac_linker_flags "-stdlib=libc++")

# Test-only AddressSanitizer build of the runtime; see x64-linux.cmake for the
# rationale. Enabled with QVAC_FABRIC_ASAN=1. Never published.
if(DEFINED ENV{QVAC_FABRIC_ASAN} AND NOT "$ENV{QVAC_FABRIC_ASAN}" STREQUAL "0")
  string(APPEND _qvac_c_flags " -fsanitize=address -fno-omit-frame-pointer")
  string(APPEND _qvac_cxx_flags " -fsanitize=address -fno-omit-frame-pointer")
  string(APPEND _qvac_linker_flags " -fsanitize=address")
endif()

set(VCPKG_C_FLAGS "${_qvac_c_flags}")
set(VCPKG_CXX_FLAGS "${_qvac_cxx_flags}")
set(VCPKG_LINKER_FLAGS "${_qvac_linker_flags}")
