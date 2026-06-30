#include <string>

#include <gtest/gtest.h>

#include "model-interface/LlamaModel.hpp"
#include "utils/BackendSelection.hpp"

using backend_selection::BackendType;

// LlamaModel::mmprojUseGpuForBackend mirrors the Android vision-encoder gating
// in commonParamsParse (QVAC-21297): the projector (mmproj/clip) runs on the
// GPU only for an OpenCL (Adreno) backend on Android; off-Android any GPU
// backend qualifies; a CPU backend never does. chooseBackend lower-cases the
// backend name (e.g. "gpuopencl", "vulkan0"), so the cases below use lower-cased
// names — matching what the production caller passes.

namespace {

TEST(MmprojGpuGating, Android_GpuOpenCl_UsesGpu) {
  // Adreno OpenCL: qvac-fabric >= 9341 runs the SigLIP / Qwen3-VL vision graph
  // correctly, so the projector runs on the GPU.
  EXPECT_TRUE(LlamaModel::mmprojUseGpuForBackend(
      BackendType::GPU, "gpuopencl", /*isAndroid=*/true));
}

TEST(MmprojGpuGating, Android_GpuVulkan_KeepsCpu) {
  // Vulkan on Android regresses SigLIP accuracy on Adreno, so the projector
  // stays on the CPU.
  EXPECT_FALSE(LlamaModel::mmprojUseGpuForBackend(
      BackendType::GPU, "vulkan0", /*isAndroid=*/true));
}

TEST(MmprojGpuGating, Android_Cpu_KeepsCpu) {
  EXPECT_FALSE(LlamaModel::mmprojUseGpuForBackend(
      BackendType::CPU, "none", /*isAndroid=*/true));
}

TEST(MmprojGpuGating, NonAndroid_AnyGpu_UsesGpu) {
  // Off-Android, any GPU backend runs the projector on the GPU (Vulkan, Metal,
  // CUDA, OpenCL).
  EXPECT_TRUE(LlamaModel::mmprojUseGpuForBackend(
      BackendType::GPU, "vulkan0", /*isAndroid=*/false));
  EXPECT_TRUE(LlamaModel::mmprojUseGpuForBackend(
      BackendType::GPU, "gpuopencl", /*isAndroid=*/false));
}

TEST(MmprojGpuGating, NonAndroid_Cpu_KeepsCpu) {
  EXPECT_FALSE(LlamaModel::mmprojUseGpuForBackend(
      BackendType::CPU, "none", /*isAndroid=*/false));
}

}  // namespace
