find_package(GTest CONFIG REQUIRED)

if(NOT DEFINED ENABLE_COVERAGE)
  set(ENABLE_COVERAGE OFF)
endif()

add_executable(
  addon-test
  test_addon_cpp.cpp
  test_async_weights_loader.cpp
  test_backend_selection.cpp
  test_cancel_stale_flag.cpp
  test_bert_model.cpp
  test_llama_lazy_init_backend.cpp
  test_logging.cpp
  test_model_metadata.cpp
  test_utils.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/AsyncWeightsLoader.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/BackendSelection.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/BertModel.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/LlamaLazyInitializeBackend.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/ModelMetadata.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/logging.cpp
  ${CMAKE_SOURCE_DIR}/addon/src/model-interface/utils.cpp
)

# Windows: define the llama/common data globals locally; they cannot be imported
# across the delay-loaded fabric module (see the shim for details).
if(WIN32)
  target_sources(addon-test PRIVATE ${CMAKE_SOURCE_DIR}/addon/src/windows_fabric_data.cpp)
endif()

target_compile_options(
  addon-test
  PRIVATE
    -Wno-deprecated
    -Wfatal-errors
    -g)

if(NOT WIN32)
  target_compile_options(
    addon-test
    PRIVATE
      -fno-omit-frame-pointer
  )
endif()

target_include_directories(
  addon-test
  PRIVATE
  ${CMAKE_SOURCE_DIR}/addon/src
  ${QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS}
)

target_compile_features(addon-test PRIVATE cxx_std_20)

if(APPLE AND NOT IOS)
  set_target_properties(addon-test PROPERTIES
    XCODE_ATTRIBUTE_MACOSX_DEPLOYMENT_TARGET "14.0"
    OSX_DEPLOYMENT_TARGET "14.0"
  )
endif()

get_filename_component(TEST_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" ABSOLUTE)
target_compile_definitions(addon-test PRIVATE 
  TEST_BINARY_DIR="${TEST_BINARY_DIR}"
)

if(ENABLE_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    MESSAGE(STATUS "ENABLING_PROFILING")

    target_compile_options(addon-test PRIVATE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(addon-test PRIVATE -fprofile-instr-generate)
  elseif(MSVC)
    message(WARNING "ENABLE_COVERAGE is not supported on MSVC in this script.")
  endif()
endif()

target_link_libraries(
  addon-test
  PRIVATE
    qvac-fabric::headers
    ${qvac_fabric_target}_module
    GTest::gtest
    GTest::gtest_main
    GTest::gmock_main
)

if(WIN32)
  # The fabric runtime is delay-loaded: the imported prebuilt module target
  # propagates /DELAYLOAD:qvac__fabric@0.bare but not the delay-load helper.
  # add_bare_module() attaches that helper (bare_delay_load -> delayimp, which
  # provides __delayLoadHelper2) to real .bare modules, but this standalone test
  # executable is a plain add_executable() and never goes through it, so it must
  # link the helper itself. The parent CMakeLists defines bare_delay_load via its
  # add_bare_module(embed-llamacpp) call before this subdirectory is added.
  target_link_libraries(
    addon-test
    PRIVATE
      bare_delay_load
  )
endif()

# Copy the shared @qvac/fabric runtime next to the test executable so it
# resolves at runtime via the $ORIGIN/@loader_path rpath set below.
if(NOT FABRIC_STATIC_MOBILE)
  add_custom_command(TARGET addon-test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:${qvac_fabric_target}_module>
      ${CMAKE_CURRENT_BINARY_DIR}/qvac__fabric@0.bare
    COMMENT "Copying qvac__fabric@0.bare to test directory"
  )

  # On desktop Linux @qvac/fabric builds its ggml compute backends as separate
  # shared libraries (GGML_BACKEND_DL): the runtime-dispatched CPU micro-arch
  # variants (and Vulkan) are dlopen'd at runtime via
  # ggml_backend_load_all_from_path(). The tests set backendsDir to this
  # directory (TEST_BINARY_DIR) and addon-test does not define BACKENDS_SUBDIR,
  # so the addon loads backends straight from here -- stage them next to the
  # executable, mirroring the production install in the parent CMakeLists.
  # Without them the ggml backend registry is empty and every model-loading
  # test fails with "no backends are loaded". On macOS/Windows the backends are
  # static inside qvac__fabric@0.bare, so the glob is empty and the CPU backend
  # self-registers when the runtime loads.
  bare_target(host)
  file(GLOB _fabric_test_backends
    "${CMAKE_SOURCE_DIR}/node_modules/@qvac/fabric/prebuilds/${host}/qvac__fabric/*.so")
  if(_fabric_test_backends)
    add_custom_command(TARGET addon-test POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${_fabric_test_backends}
        ${CMAKE_CURRENT_BINARY_DIR}/
      COMMENT "Staging @qvac/fabric ggml backends next to addon-test")
  endif()
endif()

# The runtime is named qvac__fabric@0.bare (the module SONAME) only in the copy
# beside the test binary, so point the executable's rpath at its own directory.
# Unlike the .bare addon module (which cmake-bare gives an $ORIGIN rpath), this
# plain test executable gets no such rpath by default.
if(APPLE)
  set_target_properties(addon-test PROPERTIES BUILD_RPATH "@loader_path")
elseif(NOT WIN32)
  set_target_properties(addon-test PROPERTIES BUILD_RPATH "$ORIGIN")
endif()

add_test(NAME EmbedModelTests COMMAND addon-test)
set_tests_properties(EmbedModelTests PROPERTIES
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  TIMEOUT 900)

