# TEMP (QVAC-21544): build qvac-lib-inference-addon-cpp from the JsLogger
# teardown fix (PR #2932, commit f7fe267) via an overlay so a tmp-* GPR build
# can be tested in Keet before the registry rollout.
#
# Uses vcpkg_from_github (content-addressed tarball) rather than the registry
# port's vcpkg_from_git: the parallel per-platform prebuild jobs share the
# self-hosted runner's vcpkg downloads dir, and concurrent vcpkg_from_git clones
# collide on git-tmp/.git/shallow.lock. The tarball download is lock-free and
# deduplicated by SHA512.
vcpkg_from_github(
  OUT_SOURCE_PATH SOURCE_PATH
  REPO tetherto/qvac
  REF f7fe267c91401336fd0554c6fb263c2aee9add34
  SHA512 5e17a2970b0807807d647853ebd92a850d0dcf9ed98af9bb3de245a271f62ffec94fa967309b4635b1093b6665badd87367aa5662d73ae3bd77316d0eb75aba8
)

vcpkg_check_features(
  OUT_FEATURE_OPTIONS FEATURE_OPTIONS
  FEATURES
    tests BUILD_TESTING
)

set(SOURCE_PATH "${SOURCE_PATH}/packages/inference-addon-cpp")

vcpkg_cmake_configure(
  SOURCE_PATH "${SOURCE_PATH}"
  DISABLE_PARALLEL_CONFIGURE
  OPTIONS
    ${FEATURE_OPTIONS}
)

vcpkg_cmake_install()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright
)
