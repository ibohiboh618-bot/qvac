/**
 * Pre-downloads every model the mobile bootstrap pulls so that CI can ship
 * them to AWS Device Farm as an EXTERNAL_DATA upload (see
 * `.github/workflows/test-android-sdk.yml` / `test-ios-sdk.yml`).
 *
 * Usage (matches the desktop pre-cache pattern):
 *   QVAC_CONFIG_PATH=/abs/path/to/qvac.config.json \
 *     bun run scripts/download-mobile-models.ts
 *
 * IMPORTANT — keep the constant list below in sync with the
 * `resources.define(...)` calls in `tests/mobile/consumer.ts`. We can't
 * import the consumer directly here because it pulls in `react-native`
 * and `expo-file-system`, which crash under Node/Bun. The list is the
 * full set of model constants the mobile consumer references (model
 * `constant` fields + every secondary `*Src` referenced inside their
 * configs, e.g. `vadModelSrc`, `pivotModel.modelSrc`,
 * `projectionModelSrc`, the full TTS/Parakeet shards, ...). When you
 * add a new model on the mobile consumer side, add its constant here
 * too — CI will redownload it and ship it inside EXTERNAL_DATA.
 *
 * `skipPreLoad: true` avoids the `loadModel/unloadModel` warm-up pass
 * that `downloadAllOnce` does on device — that path requires a Bare/RN
 * host, which a CI Node/Bun process doesn't have. We only need the
 * downloaded blobs on disk; loading is the device's job at test time.
 */
import {
  LLAMA_3_2_1B_INST_Q4_0,
  GTE_LARGE_FP16,
  WHISPER_TINY,
  VAD_SILERO_5_1_2,
  QWEN3_1_7B_INST_Q4,
  OCR_LATIN_RECOGNIZER_1,
  BERGAMOT_EN_FR,
  BERGAMOT_EN_ES,
  BERGAMOT_ES_EN,
  BERGAMOT_EN_IT,
  MARIAN_EN_HI_INDIC_200M_Q4_0,
  MARIAN_HI_EN_INDIC_200M_Q4_0,
  TTS_TOKENIZER_EN_CHATTERBOX,
  TTS_SPEECH_ENCODER_EN_CHATTERBOX_FP32,
  TTS_EMBED_TOKENS_EN_CHATTERBOX_FP32,
  TTS_CONDITIONAL_DECODER_EN_CHATTERBOX_FP32,
  TTS_LANGUAGE_MODEL_EN_CHATTERBOX_FP32,
  TTS_SUPERTONIC2_OFFICIAL_TEXT_ENCODER_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_DURATION_PREDICTOR_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_VECTOR_ESTIMATOR_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_VOCODER_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_UNICODE_INDEXER_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_TTS_CONFIG_SUPERTONE,
  TTS_SUPERTONIC2_OFFICIAL_VOICE_STYLE_SUPERTONE,
  PARAKEET_TDT_ENCODER_INT8,
  PARAKEET_TDT_DECODER_INT8,
  PARAKEET_TDT_PREPROCESSOR_INT8,
  PARAKEET_TDT_VOCAB,
  PARAKEET_CTC_FP32,
  PARAKEET_CTC_TOKENIZER,
  PARAKEET_SORTFORMER_FP32,
  SMOLVLM2_500M_MULTIMODAL_Q8_0,
  MMPROJ_SMOLVLM2_500M_MULTIMODAL_Q8_0,
  SALAMANDRATA_2B_INST_Q4,
  AFRICAN_4B_TRANSLATION_Q4_K_M,
  SD_V2_1_1B_Q8_0,
  downloadAsset,
  close,
} from "@qvac/sdk";
import type { ModelConstant } from "@qvac/sdk";

/**
 * Every model referenced by `tests/mobile/consumer.ts` — both top-level
 * `constant` fields and every secondary `*Src` inside the configs.
 *
 * Excluded on purpose:
 *   - `GTE_LARGE_335M_FP16_SHARD` — `skipPreDownload: true` in the
 *     consumer; the sharded-embeddings test exercises the network
 *     download path itself, so we must not seed it.
 */
const MOBILE_MODELS: readonly ModelConstant[] = [
  LLAMA_3_2_1B_INST_Q4_0,
  GTE_LARGE_FP16,
  WHISPER_TINY,
  VAD_SILERO_5_1_2,
  QWEN3_1_7B_INST_Q4,
  OCR_LATIN_RECOGNIZER_1,
  // Translation (IndicTrans + Bergamot + pivots)
  MARIAN_EN_HI_INDIC_200M_Q4_0,
  MARIAN_HI_EN_INDIC_200M_Q4_0,
  BERGAMOT_EN_FR,
  BERGAMOT_EN_ES,
  BERGAMOT_ES_EN,
  BERGAMOT_EN_IT,
  SALAMANDRATA_2B_INST_Q4,
  AFRICAN_4B_TRANSLATION_Q4_K_M,
  // TTS — Chatterbox
  TTS_TOKENIZER_EN_CHATTERBOX,
  TTS_SPEECH_ENCODER_EN_CHATTERBOX_FP32,
  TTS_EMBED_TOKENS_EN_CHATTERBOX_FP32,
  TTS_CONDITIONAL_DECODER_EN_CHATTERBOX_FP32,
  TTS_LANGUAGE_MODEL_EN_CHATTERBOX_FP32,
  // TTS — Supertonic
  TTS_SUPERTONIC2_OFFICIAL_TEXT_ENCODER_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_DURATION_PREDICTOR_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_VECTOR_ESTIMATOR_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_VOCODER_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_UNICODE_INDEXER_SUPERTONE_FP32,
  TTS_SUPERTONIC2_OFFICIAL_TTS_CONFIG_SUPERTONE,
  TTS_SUPERTONIC2_OFFICIAL_VOICE_STYLE_SUPERTONE,
  // Parakeet
  PARAKEET_TDT_ENCODER_INT8,
  PARAKEET_TDT_DECODER_INT8,
  PARAKEET_TDT_PREPROCESSOR_INT8,
  PARAKEET_TDT_VOCAB,
  PARAKEET_CTC_FP32,
  PARAKEET_CTC_TOKENIZER,
  PARAKEET_SORTFORMER_FP32,
  // Vision
  SMOLVLM2_500M_MULTIMODAL_Q8_0,
  MMPROJ_SMOLVLM2_500M_MULTIMODAL_Q8_0,
  // Diffusion
  SD_V2_1_1B_Q8_0,
];

async function main() {
  const cacheDir = process.env.QVAC_MOBILE_CACHE_DIR;
  const configPath = process.env.QVAC_CONFIG_PATH;
  if (!cacheDir || !configPath) {
    console.error(
      "❌ QVAC_MOBILE_CACHE_DIR and QVAC_CONFIG_PATH must be set; the CI " +
        "step `Configure SDK cache for pre-download` writes both.",
    );
    process.exit(2);
  }

  console.log(`📦 SDK cache directory: ${cacheDir}`);
  console.log(`📦 SDK config path:     ${configPath}`);
  console.log(`📦 ${MOBILE_MODELS.length} models to pre-cache`);

  const start = Date.now();
  for (let i = 0; i < MOBILE_MODELS.length; i++) {
    const model = MOBILE_MODELS[i];
    const label = `[${i + 1}/${MOBILE_MODELS.length}] ${model.name}`;
    const t0 = Date.now();
    process.stdout.write(`${label} ... `);
    await downloadAsset({ assetSrc: model });
    const dt = ((Date.now() - t0) / 1000).toFixed(1);
    process.stdout.write(`done (${dt}s)\n`);
  }
  const elapsed = ((Date.now() - start) / 1000).toFixed(1);

  console.log(`\n✅ Mobile bootstrap models cached in ${elapsed}s`);
}

main()
  .then(async () => {
    // `downloadAsset` spawns a Bare worker (via `@qvac/sdk` node-rpc-client)
    // and keeps an IPC socket open. Without an explicit `close()` the
    // worker stays attached and Bun's event loop never drains — the CI
    // step hangs *after* logging "✅ Mobile bootstrap models cached in
    // …s" and only releases when the runner job hits its timeout.
    // Call `close()` to terminate the worker, then exit deliberately.
    try {
      await close();
    } catch (err) {
      console.warn("⚠️  SDK close() failed (continuing):", err);
    }
    process.exit(0);
  })
  .catch(async (err) => {
    console.error("❌ download-mobile-models failed:", err);
    try {
      await close();
    } catch {
      // best-effort; we're already exiting non-zero
    }
    process.exit(1);
  });
