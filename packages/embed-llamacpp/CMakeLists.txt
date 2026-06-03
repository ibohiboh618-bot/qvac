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

set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/triplets;${VCPKG_OVERLAY_TRIPLETS}")

project(embed-llamacpp C CXX)

# When the unit tests run against an AddressSanitizer-instrumented @qvac/fabric
# (QVAC_FABRIC_ASAN=1, set by the Linux cpp-tests jobs), the test executable
# must link the *shared* libc++ that fabric also links. Statically linking
# libc++ here would put a second libc++ -- with its own locale facets and RTTI
# -- in the process alongside fabric's libc++.so.1; the duplicate type_info
# makes std::use_facet throw std::bad_cast during startup. A single shared
# libc++ also lets ASan's new/delete interpose across the module boundary.
# Production prebuilds build without QVAC_FABRIC_ASAN and keep -static-libstdc++.
set(QVAC_FABRIC_ASAN OFF)
if(DEFINED ENV{QVAC_FABRIC_ASAN}
   AND NOT "$ENV{QVAC_FABRIC_ASAN}" STREQUAL "0"
   AND NOT "$ENV{QVAC_FABRIC_ASAN}" STREQUAL "")
  set(QVAC_FABRIC_ASAN ON)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_compile_options(-stdlib=libc++)
  if(QVAC_FABRIC_ASAN)
    add_link_options(-stdlib=libc++)
  else()
    add_link_options(-stdlib=libc++ -static-libstdc++)
  endif()
endif()

find_path(VCPKG_INSTALLED_PATH share/lint-cpp/.clang-format REQUIRED)
configure_file(${VCPKG_INSTALLED_PATH}/share/lint-cpp/.clang-format
               ${CMAKE_CURRENT_SOURCE_DIR}/.clang-format COPYONLY)
configure_file(${VCPKG_INSTALLED_PATH}/share/lint-cpp/.clang-tidy
               ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy COPYONLY)
configure_file(${VCPKG_INSTALLED_PATH}/share/lint-cpp/.valgrind.supp
               ${CMAKE_CURRENT_SOURCE_DIR}/.valgrind.supp COPYONLY)
configure_file(${VCPKG_INSTALLED_PATH}/tools/lint-cpp/hooks/pre-commit
               ${CMAKE_CURRENT_SOURCE_DIR}/.git/hooks/pre-commit COPYONLY)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_path(QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS "inference-addon-cpp/JsInterface.hpp")

# The llama.cpp + ggml runtime is provided by @qvac/fabric (npm dependency)
# instead of being statically linked from the qvac-fabric vcpkg port. It ships
# the headers (qvac-fabric::headers) and a single shared .bare module that this
# addon dynamically links against, so the runtime is loaded once per process.
set(qvac-fabric_DIR "${CMAKE_CURRENT_SOURCE_DIR}/node_modules/@qvac/fabric/prebuilds/share/qvac-fabric/cmake")
find_package(qvac-fabric CONFIG REQUIRED)

if(WIN32)
  add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN -DNOGDI)
endif()

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

add_bare_module(embed-llamacpp EXPORTS)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(${embed-llamacpp}_module PRIVATE -Wl,--exclude-libs,ALL)
endif()

target_sources(
  ${embed-llamacpp}
  PRIVATE
  ${PROJECT_SOURCE_DIR}/addon/src/js-interface/binding.cpp
  ${PROJECT_SOURCE_DIR}/addon/src/model-interface/utils.cpp
  ${PROJECT_SOURCE_DIR}/addon/src/model-interface/BackendSelection.cpp
  ${PROJECT_SOURCE_DIR}/addon/src/model-interface/logging.cpp
  ${PROJECT_SOURCE_DIR}/addon/src/model-interface/LlamaLazyInitializeBackend.cpp
  ${PROJECT_SOURCE_DIR}/addon/src/model-interface/BertModel.cpp
)

# Windows: define the llama/common data globals locally; they cannot be imported
# across the delay-loaded fabric module (see the shim for details).
if(WIN32)
  target_sources(${embed-llamacpp} PRIVATE ${PROJECT_SOURCE_DIR}/addon/src/windows_fabric_data.cpp)
endif()
target_include_directories(
  ${embed-llamacpp}
  PRIVATE
    ${QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/addon/src
)

if(FABRIC_STATIC_MOBILE)
  target_link_libraries(
    ${embed-llamacpp}
    PRIVATE
      qvac-fabric::qvac-fabric-static
  )
else()
  # Compile against the llama/ggml/common headers...
  target_link_libraries(
    ${embed-llamacpp}
    PRIVATE
      qvac-fabric::headers
  )
  # ...and dynamically link the shared runtime (DT_NEEDED qvac__fabric@0.bare).
  target_link_libraries(
    ${embed-llamacpp}_module
    PRIVATE
      ${qvac_fabric_target}_module
  )
endif()

target_compile_definitions(${embed-llamacpp} PUBLIC JS_LOGGER)

bare_target(bare_target_value)
bare_module_target("." unused_target NAME module_name VERSION unused_version)
set(BACKENDS_SUBDIR_VALUE "${bare_target_value}/${module_name}")
message("Building embed-llamacpp with BACKENDS_SUBDIR='${BACKENDS_SUBDIR_VALUE}'")

target_compile_definitions(${embed-llamacpp} PRIVATE BACKENDS_SUBDIR="${BACKENDS_SUBDIR_VALUE}")

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

if(BUILD_TESTING)
  find_package(GTest CONFIG REQUIRED)
  include(GoogleTest)
  enable_testing()
  
  add_subdirectory(test/unit)
endif()

if(WIN32)
  target_link_libraries(
    ${embed-llamacpp}
    PRIVATE
      msvcrt.lib
  )
endif()
