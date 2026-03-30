#include <string>
#include <unordered_map>

#include <gtest/gtest.h>

#include "handlers/SdCtxHandlers.hpp"

using namespace qvac_lib_inference_addon_sd;
using namespace qvac_errors;

namespace {

static SdCtxConfig applyOne(const std::string& key, const std::string& value) {
  SdCtxConfig cfg;
  applySdCtxHandlers(cfg, std::unordered_map<std::string, std::string>{{key, value}});
  return cfg;
}

} // namespace

TEST(SdCtxHandlers_Prediction, SupportedValuesMapAndUnknownThrows) {
  EXPECT_EQ(applyOne("prediction", "").prediction, PREDICTION_COUNT);
  EXPECT_EQ(applyOne("prediction", "auto").prediction, PREDICTION_COUNT);
  EXPECT_EQ(applyOne("prediction", "eps").prediction, EPS_PRED);
  EXPECT_EQ(applyOne("prediction", "v").prediction, V_PRED);
  EXPECT_EQ(applyOne("prediction", "edm_v").prediction, EDM_V_PRED);
  EXPECT_EQ(applyOne("prediction", "flow").prediction, FLOW_PRED);
  EXPECT_EQ(applyOne("prediction", "flux_flow").prediction, FLUX_FLOW_PRED);
  EXPECT_EQ(applyOne("prediction", "flux2_flow").prediction, FLUX2_FLOW_PRED);

  SdCtxConfig cfg;
  EXPECT_THROW(
      applySdCtxHandlers(
          cfg,
          std::unordered_map<std::string, std::string>{{"prediction", "bogus"}}),
      StatusError);
}

TEST(SdCtxHandlers_Type, SupportedValuesMapAndUnknownThrows) {
  EXPECT_EQ(applyOne("type", "").wtype, SD_TYPE_COUNT);
  EXPECT_EQ(applyOne("type", "auto").wtype, SD_TYPE_COUNT);
  EXPECT_EQ(applyOne("type", "f32").wtype, SD_TYPE_F32);
  EXPECT_EQ(applyOne("type", "f16").wtype, SD_TYPE_F16);
  EXPECT_EQ(applyOne("type", "bf16").wtype, SD_TYPE_BF16);
  EXPECT_EQ(applyOne("type", "q4_0").wtype, SD_TYPE_Q4_0);
  EXPECT_EQ(applyOne("type", "q4_1").wtype, SD_TYPE_Q4_1);
  EXPECT_EQ(applyOne("type", "q4_k").wtype, SD_TYPE_Q4_K);
  EXPECT_EQ(applyOne("type", "q5_0").wtype, SD_TYPE_Q5_0);
  EXPECT_EQ(applyOne("type", "q5_1").wtype, SD_TYPE_Q5_1);
  EXPECT_EQ(applyOne("type", "q5_k").wtype, SD_TYPE_Q5_K);
  EXPECT_EQ(applyOne("type", "q6_k").wtype, SD_TYPE_Q6_K);
  EXPECT_EQ(applyOne("type", "q8_0").wtype, SD_TYPE_Q8_0);
  EXPECT_EQ(applyOne("type", "q2_k").wtype, SD_TYPE_Q2_K);
  EXPECT_EQ(applyOne("type", "q3_k").wtype, SD_TYPE_Q3_K);

  SdCtxConfig cfg;
  EXPECT_THROW(
      applySdCtxHandlers(
          cfg,
          std::unordered_map<std::string, std::string>{{"type", "bogus"}}),
      StatusError);
}

TEST(SdCtxHandlers_FlashAttn, ShortAndLongKeysMapTrueFalseAndInvalidThrows) {
  EXPECT_TRUE(applyOne("fa", "true").flashAttn);
  EXPECT_TRUE(applyOne("flash_attn", "1").flashAttn);
  EXPECT_FALSE(applyOne("fa", "false").flashAttn);

  SdCtxConfig cfg;
  EXPECT_THROW(
      applySdCtxHandlers(
          cfg,
          std::unordered_map<std::string, std::string>{{"fa", "maybe"}}),
      StatusError);
}

TEST(SdCtxHandlers_Rng, RngAndSamplerRngSupportedValuesAndUnknownThrow) {
  EXPECT_EQ(applyOne("rng", "cpu").rngType, CPU_RNG);
  EXPECT_EQ(applyOne("rng", "cuda").rngType, CUDA_RNG);
  EXPECT_EQ(applyOne("rng", "std_default").rngType, STD_DEFAULT_RNG);

  EXPECT_EQ(applyOne("sampler_rng", "cpu").samplerRngType, CPU_RNG);
  EXPECT_EQ(applyOne("sampler_rng", "cuda").samplerRngType, CUDA_RNG);
  EXPECT_EQ(
      applyOne("sampler_rng", "std_default").samplerRngType,
      STD_DEFAULT_RNG);

  SdCtxConfig cfgA;
  EXPECT_THROW(
      applySdCtxHandlers(
          cfgA,
          std::unordered_map<std::string, std::string>{{"rng", "bogus"}}),
      StatusError);

  SdCtxConfig cfgB;
  EXPECT_THROW(
      applySdCtxHandlers(
          cfgB,
          std::unordered_map<std::string, std::string>{{"sampler_rng", "bogus"}}),
      StatusError);
}
