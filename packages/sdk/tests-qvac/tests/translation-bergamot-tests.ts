import type { TestDefinition, Expectation } from "@tetherto/qvac-test-suite";

const createBergamotTest = (
  testId: string,
  text: string,
  resource: string,
  expectation: Expectation,
  estimatedDurationMs: number = 15000,
  suites?: string[],
): TestDefinition => ({
  testId,
  params: { text, resource },
  expectation,
  ...(suites && { suites }),
  metadata: { category: "translation-bergamot", dependency: resource, estimatedDurationMs },
});

// --- EN → FR (bergamot-en-fr) ---

export const bergamotEnFrBasic = createBergamotTest(
  "translation-bergamot-en-fr-basic",
  "Hello, how are you today?",
  "bergamot-en-fr",
  { validation: "contains-any", contains: ["bonjour", "comment", "vous", "aujourd"] },
  15000,
  ["smoke"],
);

export const bergamotEnFrLongText = createBergamotTest(
  "translation-bergamot-en-fr-long-text",
  "The weather is beautiful today. I decided to go for a walk in the park. The birds are singing and the flowers are blooming.",
  "bergamot-en-fr",
  { validation: "contains-any", contains: ["temps", "parc", "oiseaux", "fleurs", "beau"] },
  20000,
);

export const bergamotEnFrShortText = createBergamotTest(
  "translation-bergamot-en-fr-short-text",
  "Thank you very much",
  "bergamot-en-fr",
  { validation: "contains-any", contains: ["merci", "beaucoup"] },
  10000,
);

export const bergamotEnFrSpecialChars = createBergamotTest(
  "translation-bergamot-en-fr-special-chars",
  "What's your name? I'm John!",
  "bergamot-en-fr",
  { validation: "contains-any", contains: ["nom", "comment", "appel"] },
);

export const bergamotEnFrQuestion = createBergamotTest(
  "translation-bergamot-en-fr-question",
  "Can you tell me where the train station is?",
  "bergamot-en-fr",
  { validation: "contains-any", contains: ["gare", "où", "dire"] },
);

export const bergamotEnFrNumbers = createBergamotTest(
  "translation-bergamot-en-fr-numbers",
  "The meeting is at 10:30. We have 25 participants.",
  "bergamot-en-fr",
  { validation: "contains-any", contains: ["réunion", "10", "25", "participant"] },
);

export const bergamotEnFrEmptyText: TestDefinition = {
  testId: "translation-bergamot-en-fr-empty-text",
  params: { text: "", resource: "bergamot-en-fr" },
  expectation: { validation: "type", expectedType: "string" },
  metadata: { category: "translation-bergamot", dependency: "bergamot-en-fr", estimatedDurationMs: 10000 },
};

export const bergamotEnFrStreaming = createBergamotTest(
  "translation-bergamot-en-fr-streaming",
  "Good morning, how are you?",
  "bergamot-en-fr",
  { validation: "contains-any", contains: ["bonjour", "comment", "allez"] },
  15000,
  ["smoke"],
);

export const bergamotEnFrStats = createBergamotTest(
  "translation-bergamot-en-fr-stats",
  "Hello world",
  "bergamot-en-fr",
  { validation: "contains-any", contains: ["bonjour", "monde"] },
);

export const bergamotEnFrBatchBasic: TestDefinition = {
  testId: "translation-bergamot-en-fr-batch-basic",
  params: { texts: ["Good morning", "Good night"], resource: "bergamot-en-fr" },
  expectation: { validation: "contains-any", contains: ["bonjour", "matin", "nuit", "bonne"] },
  metadata: { category: "translation-bergamot", dependency: "bergamot-en-fr", estimatedDurationMs: 15000 },
};

export const bergamotEnFrBatchMultiple: TestDefinition = {
  testId: "translation-bergamot-en-fr-batch-multiple",
  params: {
    texts: ["How are you?", "The weather is nice.", "Thank you.", "Goodbye."],
    resource: "bergamot-en-fr",
  },
  expectation: { validation: "contains-any", contains: ["comment", "temps", "merci", "revoir"] },
  metadata: { category: "translation-bergamot", dependency: "bergamot-en-fr", estimatedDurationMs: 20000 },
};

// --- EN → ES (bergamot-en-es) ---

export const bergamotEnEsBasic = createBergamotTest(
  "translation-bergamot-en-es-basic",
  "Hello, how are you today?",
  "bergamot-en-es",
  { validation: "contains-any", contains: ["hola", "cómo", "estás", "hoy"] },
  15000,
  ["smoke"],
);

export const bergamotEnEsLongText = createBergamotTest(
  "translation-bergamot-en-es-long-text",
  "The weather is beautiful today. I decided to go for a walk in the park.",
  "bergamot-en-es",
  { validation: "contains-any", contains: ["tiempo", "parque", "paseo", "hermoso"] },
  20000,
);

export const bergamotEnEsQuestion = createBergamotTest(
  "translation-bergamot-en-es-question",
  "Where is the nearest hospital?",
  "bergamot-en-es",
  { validation: "contains-any", contains: ["hospital", "dónde", "cercano"] },
);

export const bergamotEnEsStreaming = createBergamotTest(
  "translation-bergamot-en-es-streaming",
  "Good morning, how are you?",
  "bergamot-en-es",
  { validation: "contains-any", contains: ["buenos", "días", "cómo"] },
);

// --- EN → IT (bergamot-en-it) ---
//
// Direct EN→IT coverage was missing from this suite — `BERGAMOT_EN_IT` was
// only exercised as the pivot leg of `bergamot-es-it-pivot`, so the consumer
// path that loads the EN→IT bergamot model directly went unchecked. This
// matches a real consumer (Keet) wallet/translation integration that
// regressed when the wrong Bergamot variant (tiny vs base-memory) shipped
// behind the registry path, producing detokenised output with whitespace
// before terminal punctuation ("Ciao mondo !" instead of "Ciao mondo!").
//
// `bergamotEnItNoSpacedPunctuation` below is the regression guard: a regex
// expectation with a negative lookahead that fails the moment any space
// precedes `!`, `?`, `.`, or `,` in the output. Word-level `contains-any`
// validations elsewhere in this file are punctuation-blind and would not
// catch this class of detokenisation regression.

export const bergamotEnItBasic = createBergamotTest(
  "translation-bergamot-en-it-basic",
  "Hello, how are you today?",
  "bergamot-en-it",
  { validation: "contains-any", contains: ["ciao", "come", "stai", "oggi"] },
  15000,
  ["smoke"],
);

export const bergamotEnItLongText = createBergamotTest(
  "translation-bergamot-en-it-long-text",
  "The weather is beautiful today. I decided to go for a walk in the park.",
  "bergamot-en-it",
  { validation: "contains-any", contains: ["tempo", "parco", "passeggiata", "bello"] },
  20000,
);

export const bergamotEnItStreaming = createBergamotTest(
  "translation-bergamot-en-it-streaming",
  "Good morning, how are you?",
  "bergamot-en-it",
  { validation: "contains-any", contains: ["buongiorno", "come", "stai"] },
);

export const bergamotEnItNoSpacedPunctuation: TestDefinition = {
  testId: "translation-bergamot-en-it-no-spaced-punctuation",
  params: { text: "Hello world!", resource: "bergamot-en-it" },
  expectation: {
    validation: "regex",
    pattern: "^(?!.*\\s[!?.,]).+$",
  },
  metadata: {
    category: "translation-bergamot",
    dependency: "bergamot-en-it",
    estimatedDurationMs: 15000,
  },
};

// --- ES → IT via EN pivot (bergamot-es-it-pivot) ---

export const bergamotPivotBasic = createBergamotTest(
  "translation-bergamot-pivot-basic",
  "Era una mañana soleada cuando María decidió visitar el mercado local.",
  "bergamot-es-it-pivot",
  { validation: "contains-any", contains: ["mattina", "sole", "maria", "mercato", "locale", "visita"] },
  30000,
);

export const bergamotPivotStreaming = createBergamotTest(
  "translation-bergamot-pivot-streaming",
  "Buenos días, ¿cómo estás hoy?",
  "bergamot-es-it-pivot",
  { validation: "contains-any", contains: ["buon", "giorno", "come", "stai", "oggi"] },
  30000,
);

export const translationBergamotTests = [
  // EN → FR
  bergamotEnFrBasic,
  bergamotEnFrLongText,
  bergamotEnFrShortText,
  bergamotEnFrSpecialChars,
  bergamotEnFrQuestion,
  bergamotEnFrNumbers,
  bergamotEnFrEmptyText,
  bergamotEnFrStreaming,
  bergamotEnFrStats,
  bergamotEnFrBatchBasic,
  bergamotEnFrBatchMultiple,
  // EN → ES
  bergamotEnEsBasic,
  bergamotEnEsLongText,
  bergamotEnEsQuestion,
  bergamotEnEsStreaming,
  // EN → IT (matches Keet consumer path; no-spaced-punctuation is the
  // regression guard for the tiny-variant Bergamot detokenisation bug)
  bergamotEnItBasic,
  bergamotEnItLongText,
  bergamotEnItStreaming,
  bergamotEnItNoSpacedPunctuation,
  // ES → IT via EN pivot
  bergamotPivotBasic,
  bergamotPivotStreaming,
];
