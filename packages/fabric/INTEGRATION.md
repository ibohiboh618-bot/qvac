# Integrating @qvac/fabric into a Consumer Addon

This guide covers the steps needed for a llama.cpp/ggml-based consumer addon to
depend on and use `@qvac/fabric` instead of statically linking the `qvac-fabric`
vcpkg port. It uses [`@qvac/llm-llamacpp`](../llm-llamacpp) and
[`@qvac/embed-llamacpp`](../embed-llamacpp) as concrete reference
implementations.

## Overview

`@qvac/fabric` is distributed as an **npm package** (bare addon). It ships
everything a consumer addon needs to build against the forked llama.cpp + ggml
runtime:

- **C++ headers** (`prebuilds/include/`) — `ggml*.h`, `gguf.h` at the include
  root; `llama.h`, `llama-cpp.h`, `common/*.h`, `mtmd/*.h` under `include/llama/`
- **CMake config** (`prebuilds/share/qvac-fabric/`) — `find_package(qvac-fabric)`
  exposes:
  - `qvac-fabric::headers` — compile-time headers (always available); adds both
    `include/` and `include/llama/` to the include path
  - `qvac-fabric::qvac-fabric-static` — static llama/ggml linking (mobile builds
    only, when `prebuilds/share/llama/` exists)
- **Prebuilt `.bare` shared library** (`prebuilds/<platform>/qvac__fabric.bare`)
  — exports the `llama_* / LLAMA_* / ggml_* / gguf_* / mtmd_*` C API and the
  `common_*` / `json_schema_to_grammar` C++ symbols; desktop consumers
  dynamically link against this
- **ggml compute backends** — shipped where ggml builds them as shared libraries
  (e.g. Android), under `prebuilds/<platform>/qvac__fabric/`

### Desktop vs Mobile

- **Desktop** (Linux, macOS, Windows): consumer addons dynamically link against
  `qvac__fabric@0.bare` via `include_bare_module`. llama/ggml/common symbols
  resolve at runtime from the shared `.bare`, so the runtime is loaded once per
  process. The ggml compute backends are static inside the shared `.bare` and
  self-register on load.
- **Mobile** (Android, iOS): controlled by the `MOBILE_DYNAMIC_LINK` CMake
  option (default `ON`). When `ON`, mobile uses the same dynamic linking as
  desktop. When `OFF`, consumer addons statically link via
  `qvac-fabric::qvac-fabric-static`.

Consumer addons do **not** need `qvac-fabric` in their own `vcpkg.json`. The
runtime comes bundled with `@qvac/fabric`.

---

## Step 1 — npm dependency

Add `@qvac/fabric` to the consumer's `package.json`:

```json
{
  "dependencies": {
    "@qvac/fabric": "^0.1.0"
  },
  "devDependencies": {
    "cmake-bare": "^1.7.5",
    "cmake-vcpkg": "^1.1.0"
  }
}
```

After `npm install`, the headers, prebuilt `.bare`, ggml backends, and cmake
configs are available under `node_modules/@qvac/fabric/prebuilds/`.

---

## Step 2 — vcpkg manifest (`vcpkg.json`)

Remove the `qvac-fabric` dependency (and the `vk-profiling` feature that pulled
its `force-profiler` feature — that now lives in `@qvac/fabric`). Keep only the
addon's own dependencies:

```json
{
  "dependencies": [
    { "name": "opencl", "platform": "android" },
    "picojson",
    "nlohmann-json",
    { "name": "qvac-lib-inference-addon-cpp", "version>=": "1.2.0" },
    { "name": "qvac-lint-cpp", "version>=": "1.4.4#3" }
  ],
  "features": {
    "tests": { "description": "Build tests", "dependencies": ["gtest"] }
  }
}
```

Do **not** add `qvac-fabric`.

---

## Step 3 — CMakeLists.txt

### Find @qvac/fabric

```cmake
option(MOBILE_DYNAMIC_LINK "Use dynamic linking for the fabric runtime on mobile" ON)

# Provides llama/ggml/common headers + the shared .bare runtime.
set(qvac-fabric_DIR "${CMAKE_CURRENT_SOURCE_DIR}/node_modules/@qvac/fabric/prebuilds/share/qvac-fabric/cmake")
find_package(qvac-fabric CONFIG REQUIRED)
```

Remove the old `find_package(llama)` / `find_package(ggml)` /
`find_package(OpenSSL)` calls and the `GGML_AVAILABLE_BACKENDS` staging loop.

### Linking — desktop vs mobile

```cmake
set(FABRIC_STATIC_MOBILE OFF)
if((ANDROID OR (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "iOS")) AND NOT MOBILE_DYNAMIC_LINK)
  set(FABRIC_STATIC_MOBILE ON)
endif()

if(NOT FABRIC_STATIC_MOBILE)
  include_bare_module("@qvac/fabric" qvac_fabric_target PREBUILD)
endif()

add_bare_module(my-consumer-addon EXPORTS)

# ... target_sources(...) / target_include_directories(...) ...

if(FABRIC_STATIC_MOBILE)
  target_link_libraries(${my-consumer-addon} PRIVATE qvac-fabric::qvac-fabric-static)
else()
  # Compile against llama/ggml/common headers...
  target_link_libraries(${my-consumer-addon} PRIVATE qvac-fabric::headers)
  # ...and dynamically link the shared runtime (DT_NEEDED qvac__fabric@0.bare).
  target_link_libraries(${my-consumer-addon}_module PRIVATE ${qvac_fabric_target}_module)
endif()
```

### Companion library + backends

The shared runtime must sit next to the consumer's `.bare` so the dynamic linker
resolves `qvac__fabric@0.bare` via RPATH. Also ship any ggml backend shared
libraries that `@qvac/fabric` staged (e.g. Android); on desktop the backends are
static inside the runtime, so the glob is empty.

```cmake
if(NOT FABRIC_STATIC_MOBILE)
  bare_target(host)
  bare_module_target("." _unused NAME addon_name)
  install(FILES $<TARGET_FILE:${qvac_fabric_target}_module>
    DESTINATION ${host}/${addon_name}
    RENAME qvac__fabric@0.bare)

  file(GLOB _fabric_backends
    "${CMAKE_CURRENT_SOURCE_DIR}/node_modules/@qvac/fabric/prebuilds/${host}/qvac__fabric/*.so")
  if(_fabric_backends)
    install(FILES ${_fabric_backends} DESTINATION ${host}/${addon_name})
  endif()
endif()
```

### How it works at runtime (desktop / dynamic mobile)

1. The consumer addon `.bare` has `DT_NEEDED: qvac__fabric@0.bare`.
2. The dynamic linker resolves it via RPATH to the companion directory.
3. If `qvac__fabric@0.bare` is already loaded (by another addon) → reuses it
   (SONAME match).
4. `llama_* / ggml_* / common_*` symbols resolve from the single loaded
   instance; the static ggml backends inside it self-register.
5. All fabric-based addons share one llama/ggml runtime in memory.

### CMake targets

| Target | Description | When available |
|--------|-------------|----------------|
| `qvac-fabric::headers` | Compile-time headers only (`include/` + `include/llama/`) | Always |
| `qvac-fabric::qvac-fabric-static` | Headers + `llama::llama;llama::common;llama::mtmd` | Mobile builds only (when `prebuilds/share/llama/` exists) |

### Symbol visibility

Consumer addons on desktop do **not** need to export the llama/ggml symbol
surface — those symbols resolve at runtime from the shared `qvac__fabric@0.bare`.
A standard consumer map exports only `bare_*` / `napi_*`:

```
{
  global:
    bare_*;
    napi_*;
  local:
    *;
};
```

The large `llama_* / ggml_* / common_*` export surface lives in `@qvac/fabric`'s
own `symbols.map` / `exports.txt`, not in the consumer.

---

## Step 4 — JS-side: pre-loading @qvac/fabric

Consumer addons that dynamically link against `qvac__fabric@0.bare` **must**
pre-load it in `binding.js` before calling `require.addon()`, so the bare runtime
has registered the `.bare` module before resolution (required for Windows
delay-load):

```js
// Pre-load @qvac/fabric so its shared .bare module (the llama.cpp + ggml
// runtime) is registered with the bare runtime before our addon triggers
// resolution of its DT_NEEDED dependency qvac__fabric@0.bare.
require('@qvac/fabric')

module.exports = require.addon()
```

---

## Step 5 — C++ usage

Includes are unchanged from a direct `find_package(llama)` build, because
`qvac-fabric::headers` exposes the same include roots:

```cpp
#include <llama.h>
#include <llama-cpp.h>
#include <ggml.h>
#include <ggml-backend.h>
#include <common/common.h>
#include <common/chat.h>
#include <common/json-schema-to-grammar.h>
```

`json_schema_to_grammar()` (from `libcommon`) takes a `nlohmann::ordered_json`,
so consumers that call it still need `nlohmann-json` in their own `vcpkg.json`
and `find_package(nlohmann_json CONFIG REQUIRED)` — only the full nlohmann
headers are required, not the runtime symbols.

The ggml backend loading path is unchanged: `LlamaLazyInitializeBackend` still
calls `ggml_backend_load_all_from_path(backendsDir / BACKENDS_SUBDIR)`. On
desktop this finds no `.so` files and the static backends (inside
`qvac__fabric@0.bare`) self-register; on Android it loads the staged backend
shared libraries.

---

## Step 6 — Build

```bash
npm install        # Resolves @qvac/fabric + devDependencies (cmake-bare, cmake-vcpkg)
npm run build      # bare-make generate && bare-make build && bare-make install
```

Verify the result:

```bash
readelf -d prebuilds/<host>/<addon>.bare | grep NEEDED   # → NEEDED qvac__fabric@0.bare
```

---

## Checklist

| # | Step | What to verify |
|---|------|----------------|
| 1 | `package.json` | `@qvac/fabric` `^0.1.0` in `dependencies`; `cmake-bare` + `cmake-vcpkg` in `devDependencies` |
| 2 | `vcpkg.json` | `qvac-fabric` is **not** listed; `vk-profiling` feature removed; addon-specific deps remain |
| 3 | `CMakeLists.txt` | `find_package(qvac-fabric ...)`; platform guard with `qvac-fabric::headers` + `include_bare_module` (desktop/dynamic) or `qvac-fabric::qvac-fabric-static` (static mobile); companion install |
| 4 | `binding.js` | `require('@qvac/fabric')` **before** `require.addon()` |
| 5 | Companion lib | Desktop: `qvac__fabric@0.bare` installed in `prebuilds/<host>/<addon>/` |
| 6 | Build | `npm run build` succeeds; `readelf -d` shows `NEEDED qvac__fabric@0.bare`; consumer `.bare` is small (no embedded ggml/llama) |
