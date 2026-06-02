// Windows-only definitions of llama.cpp `common` data globals.
//
// The qvac-fabric runtime ships as a separate shared library (qvac__fabric.bare)
// that this addon links against. On Windows, cmake-bare links sibling .bare
// modules with /DELAYLOAD and resolves them at runtime via a delay-load hook, so
// the fabric module is *delay-loaded*. Delay-loading can only patch the import
// thunks of functions; it cannot resolve imported data. That makes it impossible
// to import the `common` data globals from the fabric DLL:
//
//   * LLAMA_BUILD_NUMBER / LLAMA_COMMIT / LLAMA_COMPILER / LLAMA_BUILD_TARGET are
//     referenced by the header-scope `static std::string build_info(...)` in
//     common.h, which is instantiated in every TU that includes the header.
//   * common_log_verbosity_thold is read by the LOG_TMPL macro in log.h.
//
// We therefore define module-local copies here. ELF/Mach-O resolve such data
// across the shared runtime transparently, so this shim is only needed on
// Windows. The values are module-local: the shared runtime keeps its own copies
// for its internal use and this addon does not drive the runtime's log verbosity
// through this global.
#if defined(_WIN32)

int LLAMA_BUILD_NUMBER = 0;
const char * LLAMA_COMMIT = "unknown";
const char * LLAMA_COMPILER = "unknown";
const char * LLAMA_BUILD_TARGET = "unknown";

int common_log_verbosity_thold = 0;

#endif  // defined(_WIN32)
