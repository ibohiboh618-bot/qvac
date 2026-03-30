import type { TestDefinition, Expectation } from "@tetherto/qvac-test-suite";

const createMarianTest = (
  testId: string,
  text: string,
  resource: string,
  expectation: Expectation,
  estimatedDurationMs: number = 15000,
): TestDefinition => ({
  testId,
  params: { text, resource },
  expectation,
  metadata: { category: "translation-marian", dependency: resource, estimatedDurationMs },
});

// --- EN → ES (marian-en-es, Bergamot) ---

export const marianEnEsBasic = createMarianTest(
  "translation-marian-en-es-basic",
  "Hello, how are you today?",
  "marian-en-es",
  { validation: "contains-any", contains: ["hola", "cómo", "estás", "hoy"] },
);

export const marianEnEsLongText = createMarianTest(
  "translation-marian-en-es-long-text",
  "The weather is beautiful today. I decided to go for a walk in the park. The birds are singing and the flowers are blooming.",
  "marian-en-es",
  { validation: "contains-any", contains: ["tiempo", "parque", "pájaros", "flores", "hermoso"] },
  20000,
);

export const marianEnEsShortText = createMarianTest(
  "translation-marian-en-es-short-text",
  "Thank you very much",
  "marian-en-es",
  { validation: "contains-any", contains: ["gracias", "muchas"] },
  10000,
);

export const marianEnEsSpecialChars = createMarianTest(
  "translation-marian-en-es-special-chars",
  "What's your name? I'm John!",
  "marian-en-es",
  { validation: "contains-any", contains: ["nombre", "cómo", "llam"] },
);

export const marianEnEsQuestion = createMarianTest(
  "translation-marian-en-es-question",
  "Can you tell me where the train station is?",
  "marian-en-es",
  { validation: "contains-any", contains: ["estación", "tren", "dónde", "decir"] },
);

export const marianEnEsStreaming = createMarianTest(
  "translation-marian-en-es-streaming",
  "Good morning, how are you?",
  "marian-en-es",
  { validation: "contains-any", contains: ["buenos", "días", "cómo"] },
);

// --- ES → EN (marian-es-en) ---

export const marianEsEnBasic = createMarianTest(
  "translation-marian-es-en-basic",
  "Hola, ¿cómo estás hoy?",
  "marian-es-en",
  { validation: "contains-any", contains: ["hello", "how", "are", "you", "today"] },
);

export const marianEsEnLongText = createMarianTest(
  "translation-marian-es-en-long-text",
  "El tiempo es hermoso hoy. Decidí ir a dar un paseo por el parque. Los pájaros cantan y las flores están floreciendo.",
  "marian-es-en",
  { validation: "contains-any", contains: ["weather", "park", "birds", "flowers", "beautiful"] },
  20000,
);

export const marianEsEnQuestion = createMarianTest(
  "translation-marian-es-en-question",
  "¿Puede decirme dónde está la estación de tren?",
  "marian-es-en",
  { validation: "contains-any", contains: ["station", "train", "where", "tell"] },
);

export const translationMarianTests = [
  // EN → ES
  marianEnEsBasic,
  marianEnEsLongText,
  marianEnEsShortText,
  marianEnEsSpecialChars,
  marianEnEsQuestion,
  marianEnEsStreaming,
  // ES → EN
  marianEsEnBasic,
  marianEsEnLongText,
  marianEsEnQuestion,
];
