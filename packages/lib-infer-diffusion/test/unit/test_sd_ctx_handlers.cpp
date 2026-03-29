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
