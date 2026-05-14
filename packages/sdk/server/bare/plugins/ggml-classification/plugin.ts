import ImageClassifier from "@qvac/classification-ggml";
import {
  definePlugin,
  defineHandler,
  classifyRequestSchema,
  classifyResponseSchema,
  ModelType,
  classificationConfigSchema,
  ADDON_CLASSIFICATION,
  type ClassificationConfig,
  type CreateModelParams,
  type PluginModelResult,
} from "@/schemas";
import { createStreamLogger, registerAddonLogger } from "@/logging";
import { classify } from "@/server/bare/plugins/ggml-classification/ops/classify";

export const classificationPlugin = definePlugin({
  modelType: ModelType.ggmlClassification,
  displayName: "Image Classification (GGML)",
  addonPackage: ADDON_CLASSIFICATION,
  loadConfigSchema: classificationConfigSchema,
  skipPrimaryModelPathValidation: true,

  createModel(params: CreateModelParams): PluginModelResult {
    const config = (params.modelConfig ?? {}) as ClassificationConfig;

    const resolvedModelPath = config.modelPath ?? (params.modelPath || undefined);

    const logger = createStreamLogger(params.modelId, ModelType.ggmlClassification);
    registerAddonLogger(params.modelId, ModelType.ggmlClassification, logger);

    const model = new ImageClassifier({
      ...(resolvedModelPath ? { modelPath: resolvedModelPath } : {}),
      logger,
      nativeLogger: config.nativeLogger ?? false,
    });

    return { model };
  },

  handlers: {
    classify: defineHandler({
      requestSchema: classifyRequestSchema,
      responseSchema: classifyResponseSchema,
      streaming: true,

      handler: async function* (request) {
        const { results, modelExecutionMs } = await classify(request);
        yield {
          type: "classify" as const,
          results,
          done: true,
          modelExecutionMs,
        };
      },
    }),
  },
});
