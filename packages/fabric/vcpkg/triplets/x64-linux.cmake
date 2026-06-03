set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)
set(VCPKG_CMAKE_SYSTEM_NAME Linux)

set(VCPKG_CHAINLOAD_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/../toolchains/linux-clang.cmake")

set(_qvac_c_flags "-fPIC")
set(_qvac_cxx_flags "-fPIC -stdlib=libc++")
set(_qvac_linker_flags "-stdlib=libc++")

# Test-only AddressSanitizer build of the runtime (llama / ggml / common),
# enabled with QVAC_FABRIC_ASAN=1. Instrumenting the port gives the C++ unit
# tests full ASan visibility into the shared @qvac/fabric runtime instead of
# suppressing the malloc-vs-operator-delete mismatch an uninstrumented runtime
# produces. Build these with a dedicated binary cache dir (e.g.
# vcpkg/cache-asan) so the instrumented archives never collide with the regular
# cache. The resulting .bare is never published: it only loads inside an
# ASan-instrumented executable.
if(DEFINED ENV{QVAC_FABRIC_ASAN} AND NOT "$ENV{QVAC_FABRIC_ASAN}" STREQUAL "0")
  string(APPEND _qvac_c_flags " -fsanitize=address -fno-omit-frame-pointer")
  string(APPEND _qvac_cxx_flags " -fsanitize=address -fno-omit-frame-pointer")
  string(APPEND _qvac_linker_flags " -fsanitize=address")
endif()

set(VCPKG_C_FLAGS "${_qvac_c_flags}")
set(VCPKG_CXX_FLAGS "${_qvac_cxx_flags}")
set(VCPKG_LINKER_FLAGS "${_qvac_linker_flags}")
