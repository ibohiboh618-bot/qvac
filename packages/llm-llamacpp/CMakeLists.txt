cmake_minimum_required(VERSION 3.25)

option(ANDROID_STL "Android STL linkage" c++_shared)
option(BUILD_TESTING "Build tests" OFF)
option(ENABLE_COVERAGE "Enable coverage instrumentation for unit tests" OFF)
option(MOBILE_DYNAMIC_LINK "Use dynamic linking for the fabric runtime on mobile" ON)
if(BUILD_TESTING)
  list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()

find_package(cmake-bare REQUIRED PATHS node_modules/cmake-bare)
find_package(cmake-vcpkg REQUIRED PATHS node_modules/cmake-vcpkg)

set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/../../vcpkg-overlays/triplets;${VCPKG_OVERLAY_TRIPLETS}")

project(inference-addon-llama C CXX)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_compile_options(-stdlib=libc++)
  add_link_options(-stdlib=libc++ -static-libstdc++)
endif()

find_path(VCPKG_INSTALLED_PATH share/lint-cpp/.clang-format REQUIRED)
configure_file(${VCPKG_INSTALLED_PATH}/share/lint-cpp/.clang-format
               ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format COPYONLY)
configure_file(${VCPKG_INSTALLED_PATH}/share/lint-cpp/.clang-tidy
               ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy COPYONLY)

find_path(PICOJSON_INCLUDE_DIRS "picojson/picojson.h")
find_path(QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS "inference-addon-cpp/JsInterface.hpp")

# The llama.cpp + ggml runtime is provided by @qvac/fabric (npm dependency)
# instead of being statically linked from the qvac-fabric vcpkg port. It ships
# the headers (qvac-fabric::headers) and a single shared .bare module that this
# addon dynamically links against, so the runtime is loaded once per process.
set(qvac-fabric_DIR "${CMAKE_CURRENT_SOURCE_DIR}/node_modules/@qvac/fabric/prebuilds/share/qvac-fabric/cmake")
find_package(qvac-fabric CONFIG REQUIRED)

# Required to call llama.cpp's `json_schema_to_grammar()` for per-request
# JSON-Schema → GBNF conversion. The function signature lives in libcommon
# (exported from @qvac/fabric) but takes a `nlohmann::ordered_json`, so we need
# the full nlohmann headers, not just the forward decl shipped with
# `common/json-schema-to-grammar.h`.
find_package(nlohmann_json CONFIG REQUIRED)
# Lock-free MPMC queue (moodycamel) backing the ContinuousBatchScheduler
# request intake: `processBatch` callers enqueue, the worker dequeues.
find_package(unofficial-concurrentqueue CONFIG REQUIRED)

if(WIN32)
  add_definitions( -DNOMINMAX -DWIN32_MEAN_AND_LEAN -DNOGDI )
endif()

bare_target(bare_target_value)
bare_module_target("." unused_target NAME module_name VERSION unused_version)
set(BACKENDS_SUBDIR_VALUE "${bare_target_value}/${module_name}")
message("Building inference-addon-llama with BACKENDS_SUBDIR='${BACKENDS_SUBDIR_VALUE}'")

# Use dynamic linking (a companion qvac__fabric@0.bare) on desktop and on mobile
# unless MOBILE_DYNAMIC_LINK is OFF, in which case the runtime is statically
# linked from the libraries shipped in @qvac/fabric.
set(FABRIC_STATIC_MOBILE OFF)
if((ANDROID OR (APPLE AND CMAKE_SYSTEM_NAME STREQUAL "iOS")) AND NOT MOBILE_DYNAMIC_LINK)
  set(FABRIC_STATIC_MOBILE ON)
endif()

if(NOT FABRIC_STATIC_MOBILE)
  include_bare_module("@qvac/fabric" qvac_fabric_target PREBUILD)
endif()

add_bare_module(inference-addon-llama EXPORTS)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(${inference-addon-llama}_module PRIVATE -Wl,--exclude-libs,ALL)
endif()

  set(ADDON_SOURCES
    ${PROJECT_SOURCE_DIR}/addon/src/js-interface/binding.cpp
  )

  list(APPEND ADDON_SOURCES
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/AsyncWeightsLoader.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/CacheManager.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/ContextSlider.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/ContinuousBatchScheduler.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/GenerationParamsApply.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/LlamaLazyInitializeBackend.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/LlamaModel.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/LlamaFinetuner.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/LlamaFinetuningHelpers.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/MtmdLlmContext.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/MultiRequestBatcher.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/TextLlmContext.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/ToolsCompactController.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/ModelMetadata.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/LoggingMacros.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/BackendSelection.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/ChatTemplateUtils.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/ReasoningUtils.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/Qwen3ToolsDynamicTemplate.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/QwenTemplate.cpp
  )

  # Windows: define the llama/common data globals locally; they cannot be
  # imported across the delay-loaded fabric module (see the shim for details).
  if(WIN32)
    list(APPEND ADDON_SOURCES ${PROJECT_SOURCE_DIR}/addon/src/windows_fabric_data.cpp)
  endif()

  target_sources(
    ${inference-addon-llama}
    PRIVATE
      ${ADDON_SOURCES}
  )
  target_include_directories(
    ${inference-addon-llama}
    PRIVATE
      ${PICOJSON_INCLUDE_DIRS}
      ${QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS}
      ${PROJECT_SOURCE_DIR}/addon/src
  )
  if(FABRIC_STATIC_MOBILE)
    target_link_libraries(
      ${inference-addon-llama}
      PRIVATE
        qvac-fabric::qvac-fabric-static
        nlohmann_json::nlohmann_json
        unofficial::concurrentqueue::concurrentqueue
    )
  else()
    # Compile against the llama/ggml/common headers...
    target_link_libraries(
      ${inference-addon-llama}
      PRIVATE
        qvac-fabric::headers
        nlohmann_json::nlohmann_json
        unofficial::concurrentqueue::concurrentqueue
    )
    # ...and dynamically link the shared runtime (DT_NEEDED qvac__fabric@0.bare).
    target_link_libraries(
      ${inference-addon-llama}_module
      PRIVATE
        ${qvac_fabric_target}_module
    )
  endif()


target_compile_definitions(${inference-addon-llama} PUBLIC JS_LOGGER)
target_compile_definitions(${inference-addon-llama} PRIVATE BACKENDS_SUBDIR="${BACKENDS_SUBDIR_VALUE}")

# Install the @qvac/fabric runtime as a companion library next to this addon's
# .bare so the dynamic linker resolves qvac__fabric@0.bare via RPATH, and ship
# any ggml backend shared libraries that @qvac/fabric staged (e.g. Android). On
# desktop the backends are static inside the shared runtime, so the glob is
# empty and backends self-register when qvac__fabric@0.bare is loaded.
if(NOT FABRIC_STATIC_MOBILE)
  bare_target(host)
  bare_module_target("." _fabric_unused NAME addon_name)
  install(FILES $<TARGET_FILE:${qvac_fabric_target}_module>
    DESTINATION ${host}/${addon_name}
    RENAME qvac__fabric@0.bare)

  file(GLOB _fabric_backends
    "${CMAKE_CURRENT_SOURCE_DIR}/node_modules/@qvac/fabric/prebuilds/${host}/qvac__fabric/*.so")
  if(_fabric_backends)
    install(FILES ${_fabric_backends} DESTINATION ${host}/${addon_name})
  endif()
endif()

# Option to build CLI
option(BUILD_CLI "Build CLI tool example" FALSE)
if(BUILD_CLI)
  # Cache management integration test with automatic model download
  add_executable(cli_tool
    ${PROJECT_SOURCE_DIR}/addon/src/cli/cli_tool.cc
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/AsyncWeightsLoader.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/CacheManager.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/ContextSlider.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/ContinuousBatchScheduler.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/GenerationParamsApply.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/LlamaLazyInitializeBackend.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/LlamaModel.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/LlamaFinetuner.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/MtmdLlmContext.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/MultiRequestBatcher.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/TextLlmContext.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/ToolsCompactController.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/model-interface/ModelMetadata.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/LoggingMacros.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/BackendSelection.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/ChatTemplateUtils.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/ReasoningUtils.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/Qwen3ToolsDynamicTemplate.cpp
    ${PROJECT_SOURCE_DIR}/addon/src/utils/QwenTemplate.cpp
  )

  # Windows: define the llama/common data globals locally (see the shim).
  if(WIN32)
    target_sources(cli_tool PRIVATE ${PROJECT_SOURCE_DIR}/addon/src/windows_fabric_data.cpp)
  endif()

  target_include_directories(
    cli_tool
          PRIVATE
          ${PROJECT_SOURCE_DIR}/addon/src
          ${PICOJSON_INCLUDE_DIRS}
          ${QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS}
  )

  # Link required libraries. The CLI links the shared @qvac/fabric runtime
  # directly (it exports the full llama/ggml/common symbol surface).
  target_link_libraries(
    cli_tool
          PRIVATE
          qvac-fabric::headers
          ${qvac_fabric_target}_module
          nlohmann_json::nlohmann_json
          unofficial::concurrentqueue::concurrentqueue
  )
  find_path(QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS "inference-addon-cpp/JsInterface.hpp")
  target_include_directories(cli_tool PRIVATE ${QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS})
  # Set C++20 standard for std::views support
  target_compile_features(cli_tool PRIVATE cxx_std_20)

  # Copy the shared runtime next to the CLI binary so it resolves at runtime.
  add_custom_command(TARGET cli_tool POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:${qvac_fabric_target}_module>
      ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Copying qvac__fabric@0.bare next to cli_tool"
  )
  endif()

if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)
  enable_testing()

  # Integration tests for model classes (includes backend selection tests)
  # Pass ENABLE_COVERAGE option to test subdirectory
  add_subdirectory(test/unit)
endif()