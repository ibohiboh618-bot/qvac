#include "MobileNetGraph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <gguf.h>
#include <inference-addon-cpp/Errors.hpp>

#include "model-interface/easyocr/pipeline/qlog.hpp"

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index,readability-identifier-naming,readability-identifier-length)
// MobileNet weight loaders and graph builders use single-letter math
// identifiers, snake_case state-dict paths mirroring upstream PyTorch,
// architecture-defined layer-dim magic numbers, and raw pointer/array
// access on ggml tensor `ne[]` dimension arrays and float buffers.

namespace qvac_lib_infer_ggml_classification::graph {

namespace {

using qvac_errors::StatusError;
using qvac_errors::general_error::InternalError;
using qvac_errors::general_error::InvalidArgument;

[[noreturn]] void raise(const std::string& msg) {
  throw StatusError(InternalError, msg);
}

[[noreturn]] void raiseInvalid(const std::string& msg) {
  throw StatusError(InvalidArgument, msg);
}

// Sizing constants for the weights / graph ggml contexts and the upper
// bound passed to ggml_new_graph_custom. Deliberately oversized; the real
// MobileNet+FPN footprint stays well below them.
constexpr int kCtxTensorOverhead = 4096;
constexpr int kMaxGraphNodes = 8192;

// FPN feature-tap indices: blocks 3, 6, 12 produce the three lateral inputs
// to the FPN (matches torchvision's MobileNetV3-Large feature-extractor).
constexpr int kFpnFeatureTap1 = 3;
constexpr int kFpnFeatureTap2 = 6;
constexpr int kFpnFeatureTap3 = 12;

void printGgufMetadataKeys(const gguf_context* gguf) {
  if (gguf == nullptr) {
    QLOG(
        qvac_lib_inference_addon_cpp::logger::Priority::DEBUG,
        "[MobileNetGraph] GGUF context is null; cannot print metadata keys");
    return;
  }

  const int64_t metadataCount = gguf_get_n_kv(gguf);
  std::ostringstream os;
  os << "[MobileNetGraph] GGUF metadata keys (" << metadataCount << "):";
  for (int64_t i = 0; i < metadataCount; ++i) {
    const char* key = gguf_get_key(gguf, i);
    os << ' ' << (key != nullptr ? key : "<null>");
  }
  QLOG(qvac_lib_inference_addon_cpp::logger::Priority::DEBUG, os.str());
}

/// Tensors whose first dim is F16 are treated as storage-only; everything
/// used in runtime math (BN-folded scale/shift, FC weights) is kept as F32
/// to avoid per-layer cast operations inside the compute graph.

/// Convert a raw FP16 weight buffer to FP32 into `out`.
void fp16ToFp32(const void* src, float* out, size_t count) {
  const auto* halfPtr = static_cast<const ggml_fp16_t*>(src);
  for (size_t i = 0; i < count; ++i) {
    out[i] = ggml_fp16_to_fp32(halfPtr[i]);
  }
}

/// Same kernel-parity padding as torchvision: p = (k - 1) / 2 keeps same-size
/// output when stride=1 and reduces by floor(H/s) when stride=2.
constexpr int samePadding(int kernel) { return (kernel - 1) / 2; }

struct GraphBuilder {
  struct ggml_context* ctx;
  // GraphBuilder is a stateless one-shot helper that never outlives its
  // caller; storing the weight map by reference avoids a deep copy on every
  // graph build.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::unordered_map<std::string, struct ggml_tensor*>& w;

  [[nodiscard]] struct ggml_tensor* t(const std::string& name) const {
    auto it = w.find(name);
    if (it == w.end()) {
      raise("Missing weight tensor at graph build time: " + name);
    }
    return it->second;
  }

  /// Activation selection: HardSwish for later blocks, ReLU for early
  /// layers, matching torchvision's MobileNetV3-Large config.
  struct ggml_tensor* activate(struct ggml_tensor* x, bool useHardswish) const {
    return useHardswish
               ? ggml_unary_inplace(ctx, x, GGML_UNARY_OP_HARDSWISH)
               : ggml_relu_inplace(ctx, x);
  }

  /// Guard against silent dtype drift in the conv chain. KleidiAI's SME
  /// conv2d dispatch fires only when {kernel, src, dst} are all the same
  /// dtype, so any conv that mixes its kernel and activation dtypes is a
  /// guaranteed fallback to the plain ggml-cpu kernel. Throw at graph-build
  /// time so the mismatch surfaces immediately instead of degrading silently.
  void requireMatchingConvDtypes(
      const struct ggml_tensor* kernelT, const struct ggml_tensor* input,
      const char* opLabel) const {
    if (kernelT->type == input->type) {
      return;
    }
    raise(
        std::string("Conv dtype mismatch at ") + opLabel + ": kernel " +
        kernelT->name + " is " + ggml_type_name(kernelT->type) +
        " but input is " + ggml_type_name(input->type) +
        " — KleidiAI dispatch requires matching dtypes.");
  }

  /// Conv2d (BN folded offline into weights+bias), optionally followed by an
  /// activation.
  struct ggml_tensor* convBnAct(
      struct ggml_tensor* x, const std::string& convPrefix,
      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
       int stride, int kernel, bool activate,
      bool useHardswish, bool convertChannelFirst = false) const {
    struct ggml_tensor* kernelT = t(convPrefix + ".weight");
    const int pad = samePadding(kernel);
    requireMatchingConvDtypes(kernelT, x, "ggml_conv_2d_direct");
    if(convertChannelFirst)
    {
      kernelT = ggml_permute(ctx, kernelT, 1, 2, 0, 3);
      kernelT = ggml_cont(ctx, kernelT);
    }
    struct ggml_tensor* conv =
        ggml_conv_2d_direct(ctx, kernelT, x, stride, stride, pad, pad, 1, 1);
    conv = ggml_add_inplace(ctx, conv, t(convPrefix + ".bias_br"));

    if (!activate) {
      return conv;
    }
    return this->activate(conv, useHardswish);
  }

  struct ggml_tensor*
  fpnInBranch(struct ggml_tensor* input, int branchIndex) const {
    const std::string base =
        "dbnet.fpn.in_branches." + std::to_string(branchIndex);
    return convBnAct(
        input,
        base + ".0",
        /*stride=*/1,
        /*kernel=*/1,
        /*activate=*/true,
        /*useHardswish=*/false);
  }

  struct ggml_tensor* fpnUpsampleAdd(
      struct ggml_tensor* topDown, struct ggml_tensor* lateral) const {
    constexpr uint32_t upsampleMode =
        static_cast<uint32_t>(GGML_SCALE_MODE_BILINEAR) |
        static_cast<uint32_t>(GGML_SCALE_FLAG_ALIGN_CORNERS);
    struct ggml_tensor* upsampled = ggml_interpolate(
        ctx,
        topDown,
        lateral->ne[0],
        lateral->ne[1],
        lateral->ne[2],
        lateral->ne[3],
        upsampleMode);
    return ggml_add_inplace(ctx, upsampled, lateral);
  }

  struct ggml_tensor*
  fpnOutBranch(struct ggml_tensor* input, int branchIndex) const {
    constexpr std::array<int, 4> upsampleScaleFactors = {1, 2, 4, 8};
    const int upsampleScaleFactor =
        upsampleScaleFactors.at(static_cast<size_t>(branchIndex));
    const std::string base =
        "dbnet.fpn.out_branches." + std::to_string(branchIndex);
    struct ggml_tensor* output = convBnAct(
        input,
        base + ".0",
        /*stride=*/1,
        /*kernel=*/3,
        /*activate=*/true,
        /*useHardswish=*/false);

    constexpr uint32_t upsampleMode =
        static_cast<uint32_t>(GGML_SCALE_MODE_BILINEAR) |
        static_cast<uint32_t>(GGML_SCALE_FLAG_ALIGN_CORNERS);
    return ggml_interpolate(
        ctx,
        output,
        output->ne[0] * upsampleScaleFactor,
        output->ne[1] * upsampleScaleFactor,
        output->ne[2],
        output->ne[3],
        upsampleMode);
  }

  struct ggml_tensor* convTransposeBnAct(
      struct ggml_tensor* input, const std::string& convPrefix) const {
    struct ggml_tensor* kernelT = t(convPrefix + ".weight");
    requireMatchingConvDtypes(kernelT, input, "ggml_conv_transpose_2d_p0");
    struct ggml_tensor* conv =
        ggml_conv_transpose_2d_p0(ctx, kernelT, input, 2);
    conv = ggml_add_inplace(ctx, conv, t(convPrefix + ".bias_br"));
    return ggml_relu_inplace(ctx, conv);
  }

  struct ggml_tensor* probHead(struct ggml_tensor* input) const {
    struct ggml_tensor* output = convBnAct(
        input,
        "dbnet.prob_head.0",
        /*stride=*/1,
        /*kernel=*/3,
        /*activate=*/true,
        /*useHardswish=*/false);
    output =
        convTransposeBnAct(output, "dbnet.prob_head.3");
    struct ggml_tensor* probHead6Kernel = t("dbnet.prob_head.6.weight");
    requireMatchingConvDtypes(
        probHead6Kernel, output, "ggml_conv_transpose_2d_p0");
    output = ggml_conv_transpose_2d_p0(ctx, probHead6Kernel, output, 2);
    checkNhwcLayout(output, "conv_transpose dbnet.prob_head.6", "out");
    return ggml_add_inplace(
        ctx, output, t("dbnet.prob_head.6.bias_br"));
  }

  /// Depthwise Conv2d (BN folded offline into weights+bias) + activation.
  struct ggml_tensor* dwConvBnAct(
      struct ggml_tensor* x, const std::string& convPrefix,
      int stride, int kernel,
      bool useHardswish) const {
    struct ggml_tensor* kernelT = t(convPrefix + ".weight");
    const int pad = samePadding(kernel);
    requireMatchingConvDtypes(kernelT, x, "ggml_conv_2d_dw_direct");
    struct ggml_tensor* conv = ggml_conv_2d_dw_direct(
        ctx, kernelT, x, stride, stride, pad, pad, 1, 1);
    conv = ggml_add_inplace(ctx, conv, t(convPrefix + ".bias_br"));
    return activate(conv, useHardswish);
  }

  /// Squeeze-and-excite block: global avg pool → 1x1 conv (reduce) → ReLU →
  /// 1x1 conv (expand) → HardSigmoid → element-wise multiply with input.
  struct ggml_tensor* seBlock(
      struct ggml_tensor* x, const std::string& sePrefix, int spatialHw) const {
    // Global avg pool: kernel = full spatial extent, stride = same.
    struct ggml_tensor* pooled = ggml_pool_2d(
        ctx,
        x,
        GGML_OP_POOL_AVG,
        spatialHw,
        spatialHw,
        spatialHw,
        spatialHw,
        0,
        0);

    struct ggml_tensor* fc1Kernel = t(sePrefix + ".fc1.weight");
    requireMatchingConvDtypes(fc1Kernel, pooled, "ggml_conv_2d_direct (SE fc1)");
    struct ggml_tensor* fc1 =
        ggml_conv_2d_direct(ctx, fc1Kernel, pooled, 1, 1, 0, 0, 1, 1);
    checkNhwcLayout(fc1, ("se_fc1 " + sePrefix).c_str(), "out");
    fc1 = ggml_add_inplace(ctx, fc1, t(sePrefix + ".fc1.bias_br"));
    fc1 = ggml_relu_inplace(ctx, fc1);

    struct ggml_tensor* fc2Kernel = t(sePrefix + ".fc2.weight");
    requireMatchingConvDtypes(fc2Kernel, fc1, "ggml_conv_2d_direct (SE fc2)");
    struct ggml_tensor* fc2 =
        ggml_conv_2d_direct(ctx, fc2Kernel, fc1, 1, 1, 0, 0, 1, 1);
    fc2 = ggml_add_inplace(ctx, fc2, t(sePrefix + ".fc2.bias_br"));

    struct ggml_tensor* gate =
        ggml_unary_inplace(ctx, fc2, GGML_UNARY_OP_HARDSIGMOID);
    return ggml_mul(ctx, x, gate);
  }

  /// One torchvision InvertedResidual block.
  struct ggml_tensor* invertedResidual(
      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
      struct ggml_tensor* x, const BlockConfig& cfg, int featuresIndex,
      int inputSpatialHw) const {
    const std::string base = "features." + std::to_string(featuresIndex);
    const bool hasExpand = cfg.expansionSize != cfg.inputChannels;

    int spatial = inputSpatialHw;
    struct ggml_tensor* y = x;

    int dwBlockIdx = 0;
    int seBlockIdx = -1;
    int projBlockIdx = 0;

    if (hasExpand) {
      y = convBnAct(
          y,
          base + ".block.0.0",
          /*stride=*/1,
          /*kernel=*/1,
          /*activate=*/true,
          cfg.useHardswish);
      dwBlockIdx = 1;
      if (cfg.useSe) {
        seBlockIdx = 2;
        projBlockIdx = 3;
      } else {
        projBlockIdx = 2;
      }
    } else {
      dwBlockIdx = 0;
      if (cfg.useSe) {
        seBlockIdx = 1;
        projBlockIdx = 2;
      } else {
        projBlockIdx = 1;
      }
    }

    // Depthwise.
    const std::string dwPrefix = base + ".block." + std::to_string(dwBlockIdx);
    y = dwConvBnAct(
        y,
        dwPrefix + ".0",
        cfg.stride,
        cfg.depthwiseKernel,
        cfg.useHardswish);
    if (cfg.stride == 2) {
      spatial = (spatial + 1) / 2;
    }

    // Squeeze-and-excite.
    if (cfg.useSe) {
      const std::string sePrefix =
          base + ".block." + std::to_string(seBlockIdx);
      y = seBlock(y, sePrefix, spatial);
    }

    // Project (no activation on the tail conv).
    const std::string projPrefix =
        base + ".block." + std::to_string(projBlockIdx);
    y = convBnAct(
        y,
        projPrefix + ".0",
        /*stride=*/1,
        /*kernel=*/1,
        /*activate=*/false,
        cfg.useHardswish);

    if (cfg.stride == 1 && cfg.inputChannels == cfg.outputChannels) {
      y = ggml_add_inplace(ctx, y, x);
    }
    return y;
  }
};

} // namespace

WeightsBundle::~WeightsBundle() { reset(); }

WeightsBundle::WeightsBundle(WeightsBundle&& other) noexcept
    : ctx(std::move(other.ctx)), extraCtx(std::move(other.extraCtx)),
      tensors(std::move(other.tensors)),
      backendBuffer(other.backendBuffer),
      extraBackendBuffer(other.extraBackendBuffer),
      auxBuffers(std::move(other.auxBuffers)) {
  other.backendBuffer = nullptr;
  other.extraBackendBuffer = nullptr;
}

WeightsBundle& WeightsBundle::operator=(WeightsBundle&& other) noexcept {
  if (this != &other) {
    reset();
    ctx = std::move(other.ctx);
    extraCtx = std::move(other.extraCtx);
    tensors = std::move(other.tensors);
    backendBuffer = other.backendBuffer;
    extraBackendBuffer = other.extraBackendBuffer;
    auxBuffers = std::move(other.auxBuffers);
    other.backendBuffer = nullptr;
    other.extraBackendBuffer = nullptr;
  }
  return *this;
}

void WeightsBundle::reset() {
  tensors.clear();
  ctx.reset();
  extraCtx.reset();
  if (backendBuffer != nullptr) {
    ggml_backend_buffer_free(backendBuffer);
    backendBuffer = nullptr;
  }
  if (extraBackendBuffer != nullptr) {
    ggml_backend_buffer_free(extraBackendBuffer);
    extraBackendBuffer = nullptr;
  }
  for (ggml_backend_buffer_t buf : auxBuffers) {
    if (buf != nullptr) {
      ggml_backend_buffer_free(buf);
    }
  }
  auxBuffers.clear();
}

ComputeGraph::~ComputeGraph() { reset(); }

ComputeGraph::ComputeGraph(ComputeGraph&& other) noexcept
    : ctx(std::move(other.ctx)), graph(other.graph), allocr(other.allocr),
      input(other.input), output_1(other.output_1), output_2(other.output_2),
      output_3(other.output_3), output_4(other.output_4),
      backendBuffer(other.backendBuffer) {
  other.graph = nullptr;
  other.allocr = nullptr;
  other.input = nullptr;
  other.output_1 = nullptr;
  other.output_2 = nullptr;
  other.output_3 = nullptr;
  other.output_4 = nullptr;
  other.backendBuffer = nullptr;
}

ComputeGraph& ComputeGraph::operator=(ComputeGraph&& other) noexcept {
  if (this != &other) {
    reset();
    ctx = std::move(other.ctx);
    graph = other.graph;
    allocr = other.allocr;
    input = other.input;
    output_1 = other.output_1;
    output_2 = other.output_2;
    output_3 = other.output_3;
    output_4 = other.output_4;
    backendBuffer = other.backendBuffer;
    other.graph = nullptr;
    other.allocr = nullptr;
    other.input = nullptr;
    other.output_1 = nullptr;
    other.output_2 = nullptr;
    other.output_3 = nullptr;
    other.output_4 = nullptr;
    other.backendBuffer = nullptr;
  }
  return *this;
}

void ComputeGraph::reset() {
  graph = nullptr;
  if (allocr != nullptr) {
    ggml_gallocr_free(allocr);
    allocr = nullptr;
  }
  input = nullptr;
  output_1 = nullptr;
  output_2 = nullptr;
  output_3 = nullptr;
  output_4 = nullptr;
  ctx.reset();
  if (backendBuffer != nullptr) {
    ggml_backend_buffer_free(backendBuffer);
    backendBuffer = nullptr;
  }
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
WeightsBundle loadWeights(
    const std::string& ggufPath, std::vector<ggml_backend_t>& backends,
    std::vector<std::string>& outLabels) {
  outLabels.clear();

  // -------------------------------------------------------------------------
  // Pattern B loader (single ggml context, canonical ggml usage).
  //
  // 1. Parse the GGUF header with no_alloc=true so we get only tensor
  //    metadata in a single ggml_context — no temporary data buffers.
  // 2. Mutate any tensor types we want to override (none today, but the
  //    seam is here if a future change needs F16->F32 promotion: just
  //    flip `t->type = GGML_TYPE_F32` and recompute `nb[]` BEFORE the
  //    backend alloc).
  // 3. Synthesise the .bias_br broadcast bias tensors as new tensors in
  //    the same context — these don't exist in the GGUF.
  // 4. ONE backend allocation covers every tensor in the ctx.
  // 5. Upload pass reads raw bytes from the GGUF file at each tensor's
  //    offset, converting on the fly where in-memory dtype differs from
  //    on-disk dtype.
  //
  // No parallel ggml_context, no cloneRaw, no double-declare-then-copy.
  // -------------------------------------------------------------------------

  WeightsBundle bundle;

  // (1) GGUF header parse -> single ctx with tensor metadata only.
  struct ggml_context* ctx = nullptr;
  gguf_init_params params{.no_alloc = true, .ctx = &ctx};
  gguf_context* gguf = gguf_init_from_file(ggufPath.c_str(), params);
  if (gguf == nullptr) {
    raiseInvalid("Failed to open GGUF file: " + ggufPath);
  }
  std::unique_ptr<gguf_context, decltype(&gguf_free)> ggufGuard(
      gguf, gguf_free);
  bundle.ctx = std::unique_ptr<struct ggml_context, decltype(&ggml_free)>(
      ctx, ggml_free);
  printGgufMetadataKeys(gguf);

  // Pre-create the synthesised-tensors ctx with headroom for ~256 bias_br
  // entries. The GGUF ctx is fixed-size, so synthesised tensors must live
  // in a separate ctx.
  constexpr size_t kBiasBrHeaderBudget = 256;
  struct ggml_init_params extraParams{
      .mem_size = kBiasBrHeaderBudget * ggml_tensor_overhead(),
      .mem_buffer = nullptr,
      .no_alloc = true,
  };
  bundle.extraCtx =
      std::unique_ptr<struct ggml_context, decltype(&ggml_free)>(
          ggml_init(extraParams), ggml_free);
  if (bundle.extraCtx == nullptr) {
    raise("Failed to allocate extra ggml context for bias_br tensors");
  }
  struct ggml_context* extraCtx = bundle.extraCtx.get();

  // TODO Load all of these values from the GGUF metadata
  constexpr int fpnInBranchCount = 4;
  constexpr int fpnInBranchOutChannels = 256;
  constexpr int fpnOutBranchInputChannels = 256;
  constexpr int fpnOutBranchOutChannels = 64;
  constexpr std::array<int, fpnInBranchCount> fpnInBranchInputChannels = {
      24, 40, 112, 960};

  // (2) Per-tensor dtype overrides go here. ggml_tensor is a POD with public
  //     fields; mutating type + recomputing strides is safe BEFORE backend
  //     alloc because no data pointer has been assigned yet. Example:
  //
  //     for (const std::string& n : { "dbnet.prob_head.0.weight" }) {
  //         struct ggml_tensor* t = ggml_get_tensor(ctx, n.c_str());
  //         if (t && t->type == GGML_TYPE_F16) {
  //             t->type = GGML_TYPE_F32;
  //             t->nb[0] = ggml_type_size(GGML_TYPE_F32);
  //             for (int d = 1; d < GGML_MAX_DIMS; ++d) {
  //                 t->nb[d] = t->nb[d - 1] * t->ne[d - 1];
  //             }
  //         }
  //     }
  //
  // None today.

  // (3) Register the conv weights the graph builder will reference, and
  //     add the synthesised broadcast-bias tensors to the SAME ctx so the
  //     single backend allocation covers them.
  auto& tensors = bundle.tensors;

  auto logTensorLoad = [&](const std::string& tensorName,
                           const struct ggml_tensor* tensor) {
    std::ostringstream os;
    os << "[MobileNetGraph] loading tensor: " << tensorName
       << " (type: " << ggml_type_name(tensor->type) << ", shape: [";
    const int dims = ggml_n_dims(tensor);
    for (int i = 0; i < dims; ++i) {
      if (i > 0) {
        os << ", ";
      }
      os << tensor->ne[i];
    }
    os << "])";
    QLOG(qvac_lib_inference_addon_cpp::logger::Priority::DEBUG, os.str());
  };

  auto registerConvWeight = [&](const std::string& weightName) {
    struct ggml_tensor* t = ggml_get_tensor(ctx, weightName.c_str());
    if (t == nullptr) {
      raise("Missing tensor in GGUF: " + weightName);
    }
    logTensorLoad(weightName, t);
    tensors.emplace(weightName, t);
    return t;
  };

  // Adds a [1, 1, channels, 1] F32 broadcast-bias tensor to the SAME ctx
  // as the GGUF weights. The corresponding raw bias from the GGUF (if
  // present) is read and converted in the upload pass below; otherwise the
  // tensor is zero-filled. The F32 dtype is what ggml_add expects to pair
  // with F32 activations (and is consistent with the f16+f32->f16 binary_op
  // combo for the F16 activation path).
  auto addBiasBroadcast =
      [&](const std::string& biasName, int64_t channels) {
        const std::string brName = biasName + "_br";
        if (tensors.contains(brName)) {
          return;
        }
        const std::array<int64_t, 4> shape4d = {1, 1, channels, 1};
        struct ggml_tensor* t = ggml_new_tensor(
            extraCtx, GGML_TYPE_F32, 4, shape4d.data());
        ggml_set_name(t, brName.c_str());
        logTensorLoad(brName, t);
        tensors.emplace(brName, t);
      };

  // Convenience: register a conv weight by its prefix and add its bias_br.
  // OC = ne[3] for both regular and depthwise conv weights (depthwise stores
  // [KW, KH, 1, C] with C = OC = IC).
  auto registerConvWithBias = [&](const std::string& convPrefix) {
    struct ggml_tensor* w = registerConvWeight(convPrefix + ".weight");
    addBiasBroadcast(convPrefix + ".bias", w->ne[3]);
    return w;
  };

  // Stem.
  registerConvWithBias("features.0.0");

  // 15 inverted residual blocks.
  int featureIndex = 1;
  for (const BlockConfig& cfg : kBlocks) {
    const std::string base = "features." + std::to_string(featureIndex);
    const bool hasExpand = cfg.expansionSize != cfg.inputChannels;
    int dwIdx = 0;
    int seIdx = -1;
    int projIdx = 0;
    if (hasExpand) {
      registerConvWithBias(base + ".block.0.0");
      dwIdx = 1;
      if (cfg.useSe) {
        seIdx = 2;
        projIdx = 3;
      } else {
        projIdx = 2;
      }
    } else {
      if (cfg.useSe) {
        seIdx = 1;
        projIdx = 2;
      } else {
        projIdx = 1;
      }
    }
    const std::string dwBase = base + ".block." + std::to_string(dwIdx);
    registerConvWithBias(dwBase + ".0");

    if (cfg.useSe) {
      const std::string seBase = base + ".block." + std::to_string(seIdx);
      registerConvWithBias(seBase + ".fc1");
      registerConvWithBias(seBase + ".fc2");
    }

    const std::string projBase = base + ".block." + std::to_string(projIdx);
    registerConvWithBias(projBase + ".0");

    ++featureIndex;
  }

  // Tail.
  registerConvWithBias("features.16.0");

  // FPN input branches.
  for (int branch = 0; branch < fpnInBranchCount; ++branch) {
    const std::string base = "dbnet.fpn.in_branches." + std::to_string(branch);
    struct ggml_tensor* conv = registerConvWithBias(base + ".0");
    if (conv->ne[0] != 1 || conv->ne[1] != 1 ||
        conv->ne[2] !=
            fpnInBranchInputChannels.at(static_cast<size_t>(branch)) ||
        conv->ne[3] != fpnInBranchOutChannels) {
      raise("FPN input branch conv shape mismatch for " + base + ".0.weight");
    }
  }

  // FPN output branches.
  for (int branch = 0; branch < fpnInBranchCount; ++branch) {
    const std::string base = "dbnet.fpn.out_branches." + std::to_string(branch);
    struct ggml_tensor* conv = registerConvWithBias(base + ".0");
    if (conv->ne[0] != 3 || conv->ne[1] != 3 ||
        conv->ne[2] != fpnOutBranchInputChannels ||
        conv->ne[3] != fpnOutBranchOutChannels) {
      raise("FPN output branch conv shape mismatch for " + base + ".0.weight");
    }
  }

  // DBNet probability head.
  // prob_head.0 is a regular Conv2d: output channels = ne[3]. The default
  // registerConvWithBias is correct.
  registerConvWithBias("dbnet.prob_head.0");
  // prob_head.3 and prob_head.6 are ConvTranspose2d. In GGUF their weight is
  // stored as [KW, KH, OC, IC], so the output-channel count is ne[2], NOT
  // ne[3]. Register the weight and synthesise the bias_br with the correct
  // channel count explicitly.
  {
    struct ggml_tensor* w = registerConvWeight("dbnet.prob_head.3.weight");
    addBiasBroadcast("dbnet.prob_head.3.bias", w->ne[2]);
  }
  {
    struct ggml_tensor* w = registerConvWeight("dbnet.prob_head.6.weight");
    addBiasBroadcast("dbnet.prob_head.6.bias", w->ne[2]);
  }

  // (4) Backend allocation, routed per-tensor through the CPU device's
  //     `extra_bufts` list. The first extra buft that reports a packed size
  //     larger than the raw tensor bytes (signalling it would prepack the
  //     tensor at set_tensor time) wins for plain Conv2D weights — that's how
  //     KleidiAI's SME prepack engages without our code naming
  //     `ggml_backend_cpu_kleidiai_buffer_type` directly. A future GPU
  //     backend would surface its preferred buft the same way.
  //
  //     ConvTranspose weights (`dbnet.prob_head.3.weight`,
  //     `dbnet.prob_head.6.weight`) MUST stay in the default CPU buft. The
  //     KleidiAI buft's set_tensor unconditionally prepacks any F32/F16
  //     tensor it owns (it can't tell whether the tensor will be used by
  //     GGML_OP_CONV_2D or GGML_OP_CONV_TRANSPOSE_2D). ConvTranspose ops
  //     never dispatch through KleidiAI, so the standard CPU kernel would
  //     read the mangled prepacked bytes as raw `[KW,KH,OC,IC]` and produce
  //     garbage.
  //
  //     Synthesised bias_br tensors and the unreferenced raw GGUF .bias
  //     tensors stay in the default CPU buft too — they aren't conv kernels.
  ggml_backend_buffer_type_t defaultCpuBuft =
      ggml_backend_get_default_buffer_type(backends[0]);
  ggml_backend_buffer_type_t prepackBuft = nullptr;
  {
    ggml_backend_buffer_type_t* extraBufts = nullptr;
    ggml_backend_dev_t cpuDev = ggml_backend_get_device(backends[0]);
    if (cpuDev != nullptr) {
      ggml_backend_reg_t cpuReg = ggml_backend_dev_backend_reg(cpuDev);
      if (cpuReg != nullptr) {
        auto* getExtras =
            reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
                ggml_backend_reg_get_proc_address(
                    cpuReg, "ggml_backend_dev_get_extra_bufts"));
        if (getExtras != nullptr) {
          extraBufts = getExtras(cpuDev);
        }
      }
    }
    if (extraBufts != nullptr) {
      struct ggml_tensor* probe =
          ggml_get_tensor(ctx, "features.0.0.weight");
      if (probe != nullptr) {
        for (ggml_backend_buffer_type_t* p = extraBufts; *p != nullptr; ++p) {
          if (ggml_backend_buft_get_alloc_size(*p, probe) >
              ggml_nbytes(probe)) {
            prepackBuft = *p;
            break;
          }
        }
      }
    }
  }

  // Names of ConvTranspose weights — `prob_head.3.weight` looks like a
  // regular Conv2D by shape (ne=[2,2,64,64]) so it needs an explicit name
  // entry. `prob_head.6.weight` has ne[2]=1 and is caught by the shape
  // filter below, but list it here too for documentation.
  const std::unordered_set<std::string> convTransposeWeightNames = {
      "dbnet.prob_head.3.weight",
      "dbnet.prob_head.6.weight",
  };

  // Returns true when the weight tensor is NOT a regular Conv2D kernel and
  // therefore must NOT be routed to a prepack buft. The KleidiAI buft's
  // set_tensor unconditionally prepacks F32/F16 tensors it owns, so any
  // kernel whose runtime op is not GGML_OP_CONV_2D (depthwise, transpose,
  // matmul, etc.) would get its bytes mangled into KleidiAI's NHWC layout
  // and read as garbage by the standard CPU kernel.
  //
  // Depthwise kernels in MobileNetV3-Large have ne[2]==1 (one input channel
  // per group). KleidiAI's own conv2d preconditions also reject this shape
  // (`kernel->ne[2] == src->ne[2]` would fail with src having C channels),
  // so `ne[2] == 1` is a sound "not a regular Conv2D kernel" filter for
  // this network.
  auto isPrepackable = [&](const std::string& name,
                           const struct ggml_tensor* t) {
    if (!name.ends_with(".weight")) {
      return false;
    }
    if (convTransposeWeightNames.contains(name)) {
      return false;
    }
    if (t->ne[2] == 1) {
      return false;
    }
    return true;
  };

  for (auto& [name, dst] : tensors) {
    if (!name.ends_with(".weight")) {
      continue;
    }
    if (dst->ne[2] != 1) {
      continue;
    }
    if (convTransposeWeightNames.contains(name)) {
      continue;
    }
    dst->nb[2] = ggml_type_size(dst->type);
  }

  if (prepackBuft != nullptr) {
    // Per-tensor allocation for the prepack-eligible conv weights so their
    // bytes flow through the KleidiAI buft's set_tensor hook at upload
    // (which writes the magic header + NHWC-packed payload). One buffer per
    // tensor — small surface, easy cleanup.
    for (auto& [name, dst] : tensors) {
      if (!isPrepackable(name, dst)) {
        continue;
      }
      const size_t allocSize =
          ggml_backend_buft_get_alloc_size(prepackBuft, dst);
      ggml_backend_buffer_t buf =
          ggml_backend_buft_alloc_buffer(prepackBuft, allocSize);
      if (buf == nullptr) {
        raise(
            "Failed to allocate per-tensor prepack buffer for " + name);
      }
      ggml_tallocr tallocr = ggml_tallocr_new(buf);
      if (ggml_tallocr_alloc(&tallocr, dst) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf);
        raise("ggml_tallocr_alloc failed for " + name);
      }
      bundle.auxBuffers.push_back(buf);
    }
  }

  // Everything in bundle.ctx that wasn't pre-allocated above
  // (ConvTranspose .weight, raw .bias 1D tensors, the registered conv
  // weights when prepackBuft is null) goes into one default-CPU buffer.
  bundle.backendBuffer = ggml_backend_alloc_ctx_tensors_from_buft(
      bundle.ctx.get(), defaultCpuBuft);
  if (bundle.backendBuffer == nullptr) {
    raise("Failed to allocate backend buffer for GGUF weights");
  }
  // Synthesised bias_br tensors live in extraCtx.
  bundle.extraBackendBuffer = ggml_backend_alloc_ctx_tensors_from_buft(
      bundle.extraCtx.get(), defaultCpuBuft);
  if (bundle.extraBackendBuffer == nullptr) {
    raise("Failed to allocate backend buffer for synthesised tensors");
  }

  // (5) Upload pass. Read raw bytes from the GGUF file at each tensor's
  //     offset, converting on the fly where in-memory dtype differs from
  //     on-disk dtype, and push via ggml_backend_tensor_set.
  std::ifstream file(ggufPath, std::ios::binary);
  if (!file) {
    raise("Failed to open GGUF file for raw read: " + ggufPath);
  }
  const size_t dataBaseOffset = gguf_get_data_offset(gguf);

  // Read `count` F16 elements from `fileOffset` and return them as F32.
  auto readF16AsF32 = [&](size_t fileOffset, size_t count) {
    std::vector<ggml_fp16_t> raw(count);
    file.seekg(static_cast<std::streamoff>(fileOffset));
    file.read(
        reinterpret_cast<char*>(raw.data()),
        static_cast<std::streamsize>(count * sizeof(ggml_fp16_t)));
    if (!file) {
      raise("File read failed while loading F16 tensor data");
    }
    std::vector<float> out(count);
    fp16ToFp32(raw.data(), out.data(), count);
    return out;
  };

  // Read `bytes` raw bytes from `fileOffset`.
  auto readBytes = [&](size_t fileOffset, size_t bytes) {
    std::vector<uint8_t> buf(bytes);
    file.seekg(static_cast<std::streamoff>(fileOffset));
    file.read(
        reinterpret_cast<char*>(buf.data()),
        static_cast<std::streamsize>(bytes));
    if (!file) {
      raise("File read failed while loading tensor data");
    }
    return buf;
  };

  for (auto& [name, dst] : tensors) {
    if (name.ends_with(".bias_br")) {
      // Synthesised broadcast bias: pull values from the corresponding raw
      // bias key in the GGUF (F16 or F32) and convert to F32, or fill with
      // zeros if the key is absent.
      const std::string biasName =
          name.substr(0, name.size() - std::string("_br").size());
      const int64_t biasIdx = gguf_find_tensor(gguf, biasName.c_str());
      const size_t nElements = static_cast<size_t>(ggml_nelements(dst));
      std::vector<float> values(nElements, 0.0F);
      if (biasIdx >= 0) {
        const struct ggml_tensor* biasMeta =
            ggml_get_tensor(ctx, biasName.c_str());
        if (biasMeta != nullptr &&
            static_cast<size_t>(ggml_nelements(biasMeta)) == nElements) {
          const size_t off =
              dataBaseOffset + gguf_get_tensor_offset(gguf, biasIdx);
          if (biasMeta->type == GGML_TYPE_F16) {
            values = readF16AsF32(off, nElements);
          } else if (biasMeta->type == GGML_TYPE_F32) {
            std::vector<uint8_t> raw =
                readBytes(off, nElements * sizeof(float));
            std::memcpy(values.data(), raw.data(), nElements * sizeof(float));
          } else {
            raise(
                "Unsupported bias dtype " +
                std::string(ggml_type_name(biasMeta->type)) + " for " +
                biasName);
          }
        }
      }
      ggml_backend_tensor_set(
          dst, values.data(), 0, values.size() * sizeof(float));
      continue;
    }

    // Conv weight: same name in the GGUF.
    const int64_t idx = gguf_find_tensor(gguf, name.c_str());
    if (idx < 0) {
      raise("Tensor missing from GGUF: " + name);
    }
    const size_t off = dataBaseOffset + gguf_get_tensor_offset(gguf, idx);

    // Source dtype from the parsed GGUF metadata (unchanged by us) vs the
    // destination dtype in the ctx (which we may have mutated in step 2).
    // For now they match — fast same-dtype byte copy. If a future change
    // promotes a weight to F32, add an F16->F32 branch here mirroring the
    // bias_br case above.
    const size_t bytes = ggml_nbytes(dst);
    std::vector<uint8_t> raw = readBytes(off, bytes);
    ggml_backend_tensor_set(dst, raw.data(), 0, bytes);
  }

  return bundle;
}

ComputeGraph buildGraph(
    const WeightsBundle& weights, std::vector<ggml_backend_t>& backends) {
  ComputeGraph cg;
  const size_t ctxSize =
      (ggml_tensor_overhead() * kCtxTensorOverhead) + ggml_graph_overhead();
  cg.ctx = std::unique_ptr<struct ggml_context, decltype(&ggml_free)>(
      ggml_init({.mem_size = ctxSize, .mem_buffer = nullptr, .no_alloc = true}),
      ggml_free);
  if (!cg.ctx) {
    raise("Failed to allocate graph ggml context");
  }
  struct ggml_context* ctx = cg.ctx.get();

  // WHCN order: W, H, C, N.
  cg.input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kInputHw, kInputHw, 3, 1);
  ggml_set_name(cg.input, "input");

  struct ggml_tensor* inputNhwc =
      ggml_cont(ctx, ggml_permute(ctx, cg.input, 1, 2, 0, 3));
  inputNhwc = ggml_permute(ctx, inputNhwc, 2, 0, 1, 3);
  ggml_set_name(inputNhwc, "input_nhwc");

  GraphBuilder gb{.ctx = ctx, .w = weights.tensors};

  // Stem.
  struct ggml_tensor* x = gb.convBnAct(
      inputNhwc,
      "features.0.0",
      /*stride=*/2,
      /*kernel=*/3,
      /*activate=*/true,
      /*useHardswish=*/true);

  int spatial = kInputHw / 2; // 112 after stem

  // 15 inverted residual blocks.
  int graphFeatureIndex = 1;
  for (const BlockConfig& cfg : kBlocks) {
    x = gb.invertedResidual(x, cfg, graphFeatureIndex, spatial);
    if (cfg.stride == 2) {
      spatial = (spatial + 1) / 2;
    }
    switch (graphFeatureIndex) {
    case kFpnFeatureTap1:
      cg.output_1 = x;
      break;
    case kFpnFeatureTap2:
      cg.output_2 = x;
      break;
    case kFpnFeatureTap3:
      cg.output_3 = x;
      break;
    default:
      break;
    }
    ++graphFeatureIndex;
  }

  // Tail (features.16): 1x1 conv + BN + HardSwish at 7x7 spatial.
  x = gb.convBnAct(
      x,
      "features.16.0",
      /*stride=*/1,
      /*kernel=*/1,
      /*activate=*/true,
      /*useHardswish=*/true);

  if (cg.output_1 == nullptr || cg.output_2 == nullptr ||
      cg.output_3 == nullptr) {
    raise("Missing backbone feature map for FPN input branches");
  }

  // FPN in_branches: project C2/C3/C4/C5 to 256 channels with 1x1 conv + BN +
  // ReLU.
  cg.output_1 = gb.fpnInBranch(cg.output_1, 0);
  cg.output_2 = gb.fpnInBranch(cg.output_2, 1);
  cg.output_3 = gb.fpnInBranch(cg.output_3, 2);
  cg.output_4 = gb.fpnInBranch(x, 3);

  // FPN top-down path: out = [_x[-1]]; append(upsample(out[-1]) + t)
  // for the lower-level lateral features, using bilinear align_corners=True.
  cg.output_3 = gb.fpnUpsampleAdd(cg.output_4, cg.output_3);
  cg.output_2 = gb.fpnUpsampleAdd(cg.output_3, cg.output_2);
  cg.output_1 = gb.fpnUpsampleAdd(cg.output_2, cg.output_1);

  // FPN out_branches consume the top-down outputs in low-to-high order
  // (`out[::-1]` in the PyTorch reference), then upsample to the C2 size.
  cg.output_1 = gb.fpnOutBranch(cg.output_1, 0);
  cg.output_2 = gb.fpnOutBranch(cg.output_2, 1);
  cg.output_3 = gb.fpnOutBranch(cg.output_3, 2);
  cg.output_4 = gb.fpnOutBranch(cg.output_4, 3);

  // PyTorch cats NCHW tensors on dim=1 (channels). In ggml WHCN layout the
  // channel axis is dim=2, yielding the 256-channel DBNet feature map.
  struct ggml_tensor* fpnCat12 = ggml_concat(ctx, cg.output_1, cg.output_2, 2);
  struct ggml_tensor* fpnCat34 = ggml_concat(ctx, cg.output_3, cg.output_4, 2);
  cg.output_4 = ggml_concat(ctx, fpnCat12, fpnCat34, 2);
  cg.output_4 = gb.probHead(cg.output_4);
  // cg.output_4 = ggml_sigmoid(ctx, cg.output_4);

  ggml_set_name(cg.output_1, "output_1");
  ggml_set_name(cg.output_2, "output_2");
  ggml_set_name(cg.output_3, "output_3");
  ggml_set_name(cg.output_4, "output_4");

  cg.graph = ggml_new_graph_custom(ctx, kMaxGraphNodes, /*grads=*/false);
  ggml_build_forward_expand(cg.graph, cg.output_4);

  cg.allocr =
      ggml_gallocr_new(ggml_backend_get_default_buffer_type(backends[0]));
  if (cg.allocr == nullptr) {
    raise("Failed to create graph allocator for compute graph");
  }

  if (!ggml_gallocr_alloc_graph(cg.allocr, cg.graph)) {
    raise("Failed to allocate compute graph");
  }

  if (const char* dumpPath = std::getenv("OCR_DUMP_GRAPH_DETECTOR");
      dumpPath != nullptr && dumpPath[0] != '\0') {
    // Forward-only graph: `gb` (1st arg) is the graph that gets walked and
    // must be non-null; `cgraph` (2nd arg) is only consulted for coloring
    // when gradients are present, and is null-checked, so pass nullptr.
    ggml_graph_dump_dot(cg.graph, nullptr, dumpPath);
  }

  return cg;
}

} // namespace qvac_lib_infer_ggml_classification::graph

// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,cppcoreguidelines-pro-bounds-constant-array-index,readability-identifier-naming,readability-identifier-length)
