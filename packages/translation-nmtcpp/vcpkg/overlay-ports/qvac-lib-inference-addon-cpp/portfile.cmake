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
  REF c5380ebcb4dda8647ef3d3b2917860a91dd2a3be
  SHA512 df0db0fc3d4fe914a5e2a77662cbf1370641c8e20970d2089cc415c9900e92b592f707960888bf3e9cd08d4e65a41ddbe65343bb35129d522fa625fa2cc5ab57
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
