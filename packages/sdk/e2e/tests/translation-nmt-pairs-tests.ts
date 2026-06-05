// QVAC-18959 — SDK ↔ registry ↔ model-link sanity tests for every NMT pair
// that ships a registry constant. One test per pair: load the model via the
// SDK, translate a short known phrase, and assert the output contains an
// expected token. Catches mismatches that addon-only tests miss (wrong
// constant wiring, broken model link, vocab resolution, from/to mapping).
//
// Tagged with the `nmt-langpairs` suite so the default "full" runs can opt
// out via `exclude-suite=nmt-langpairs`, and on-demand runs select via
// `filter=translation-nmt-pair` or `suite=nmt-langpairs`.

import type { TestDefinition, Expectation } from "@tetherto/qvac-test-suite";

// ---- Canonical English source ----------------------------------------------

/** Input for every en→XX pair. */
export const EN_SHORT = "Hello, how are you?";

/**
 * Lowercased substrings expected in the English output of XX→en pairs.
 * Lenient (contains-any): any one match passes. Covers the common
 * register variants bergamot produces for "Hello, how are you?" —
 * formal ("hello", "how"), informal ("hi", "hey"), and "good morning" /
 * "what's up" forms ("good", "you"). Substring match, so "you" also
 * catches "how are you" / "are you doing" without false-positive risk
 * on legitimate translation output.
 */
export const EN_KEYWORDS = ["hello", "hi", "hey", "how", "you", "good"];

// ---- Per-language data ------------------------------------------------------

interface LangEntry {
  /** Display name, used only in metadata/logs. */
  name: string;
  /** "Hello, how are you?" in this language — input for every XX→en pair. */
  short: string;
  /**
   * Lowercased substrings expected in en→XX output. ValidationHelpers
   * lowercases both sides, so non-Latin scripts (Cyrillic/CJK/Indic/Arabic)
   * use natural case here.
   */
  keywords: string[];
}

const LANGUAGES: Record<string, LangEntry> = {
  ar: { name: "Arabic",            short: "مرحبا، كيف حالك؟",                keywords: ["مرحبا", "كيف"] },
  az: { name: "Azerbaijani",       short: "Salam, necəsən?",                keywords: ["salam", "necə"] },
  be: { name: "Belarusian",        short: "Прывітанне, як справы?",          keywords: ["прывітанне", "справы"] },
  bg: { name: "Bulgarian",         short: "Здравей, как си?",                keywords: ["здравей", "как"] },
  bn: { name: "Bengali",           short: "নমস্কার, কেমন আছেন?",                keywords: ["নমস্কার", "কেমন"] },
  bs: { name: "Bosnian",           short: "Zdravo, kako si?",                keywords: ["zdravo", "kako"] },
  ca: { name: "Catalan",           short: "Hola, com estàs?",                keywords: ["hola", "com"] },
  cs: { name: "Czech",             short: "Ahoj, jak se máš?",               keywords: ["ahoj", "jak"] },
  da: { name: "Danish",            short: "Hej, hvordan har du det?",        keywords: ["hej", "hvordan"] },
  de: { name: "German",            short: "Hallo, wie geht es dir?",         keywords: ["hallo", "geht"] },
  el: { name: "Greek",             short: "Γεια σου, πώς είσαι;",            keywords: ["γεια", "πώς"] },
  es: { name: "Spanish",           short: "Hola, ¿cómo estás?",              keywords: ["hola", "cómo"] },
  et: { name: "Estonian",          short: "Tere, kuidas läheb?",             keywords: ["tere", "kuidas"] },
  fa: { name: "Persian",           short: "سلام، حال شما چطور است؟",          keywords: ["سلام", "چطور"] },
  fi: { name: "Finnish",           short: "Hei, miten voit?",                keywords: ["hei", "miten"] },
  fr: { name: "French",            short: "Bonjour, comment allez-vous?",    keywords: ["bonjour", "comment"] },
  gu: { name: "Gujarati",          short: "નમસ્તે, કેમ છો?",                    keywords: ["નમસ્તે", "કેમ", "હેલો", "કેવી"] },
  he: { name: "Hebrew",            short: "שלום, מה שלומך?",                  keywords: ["שלום", "מה"] },
  hi: { name: "Hindi",             short: "नमस्ते, आप कैसे हैं?",                keywords: ["नमस्ते", "कैसे"] },
  hr: { name: "Croatian",          short: "Bok, kako si?",                   keywords: ["bok", "kako"] },
  hu: { name: "Hungarian",         short: "Szia, hogy vagy?",                keywords: ["szia", "hogy"] },
  id: { name: "Indonesian",        short: "Halo, apa kabar?",                keywords: ["halo", "kabar"] },
  is: { name: "Icelandic",         short: "Halló, hvernig hefurðu það?",     keywords: ["halló", "hvernig"] },
  it: { name: "Italian",           short: "Ciao, come stai?",                keywords: ["ciao", "come"] },
  ja: { name: "Japanese",          short: "こんにちは、お元気ですか？",              keywords: ["こんにちは", "元気"] },
  kn: { name: "Kannada",           short: "ನಮಸ್ಕಾರ, ಹೇಗಿದ್ದೀರಿ?",                keywords: ["ನಮಸ್ಕಾರ", "ಹೇಗಿದ್ದೀರಿ"] },
  ko: { name: "Korean",            short: "안녕하세요, 어떻게 지내세요?",              keywords: ["안녕하세요", "어떻게", "여보세요", "어떠세요"] },
  lt: { name: "Lithuanian",        short: "Labas, kaip sekasi?",             keywords: ["labas", "kaip"] },
  lv: { name: "Latvian",           short: "Sveiki, kā jums klājas?",         keywords: ["sveiki", "kā"] },
  ml: { name: "Malayalam",         short: "നമസ്കാരം, സുഖമാണോ?",                keywords: ["നമസ്കാരം", "സുഖമാണോ", "ഹല", "എങ്ങന"] },
  ms: { name: "Malay",             short: "Helo, apa khabar?",               keywords: ["helo", "khabar"] },
  mt: { name: "Maltese",           short: "Bongu, kif int?",                 keywords: ["bongu", "kif"] },
  nb: { name: "Norwegian Bokmål",  short: "Hei, hvordan har du det?",        keywords: ["hei", "hvordan"] },
  nl: { name: "Dutch",             short: "Hallo, hoe gaat het?",            keywords: ["hallo", "gaat"] },
  nn: { name: "Norwegian Nynorsk", short: "Hei, korleis har du det?",        keywords: ["hei", "korleis"] },
  no: { name: "Norwegian",         short: "Hei, hvordan har du det?",        keywords: ["hei", "hvordan"] },
  pl: { name: "Polish",            short: "Cześć, jak się masz?",            keywords: ["cześć", "jak"] },
  pt: { name: "Portuguese",        short: "Olá, como vai?",                  keywords: ["olá", "como"] },
  ro: { name: "Romanian",          short: "Salut, ce mai faci?",             keywords: ["salut", "faci"] },
  ru: { name: "Russian",           short: "Привет, как дела?",                keywords: ["привет", "как"] },
  sk: { name: "Slovak",            short: "Ahoj, ako sa máš?",               keywords: ["ahoj", "ako"] },
  sl: { name: "Slovenian",         short: "Pozdravljeni, kako ste?",         keywords: ["pozdravljeni", "kako", "živjo", "kakšen"] },
  sq: { name: "Albanian",          short: "Përshëndetje, si je?",            keywords: ["përshëndetje", "si"] },
  sr: { name: "Serbian",           short: "Здраво, како си?",                 keywords: ["здраво", "како"] },
  sv: { name: "Swedish",           short: "Hej, hur mår du?",                keywords: ["hej", "mår"] },
  ta: { name: "Tamil",             short: "வணக்கம், எப்படி இருக்கிறீர்கள்?",      keywords: ["வணக்கம்", "எப்படி"] },
  te: { name: "Telugu",            short: "నమస్కారం, ఎలా ఉన్నారు?",             keywords: ["నమస్కారం", "ఎలా"] },
  th: { name: "Thai",              short: "สวัสดี, สบายดีไหม?",                keywords: ["สวัสดี", "สบายดี"] },
  tr: { name: "Turkish",           short: "Merhaba, nasılsın?",              keywords: ["merhaba", "nasılsın"] },
  uk: { name: "Ukrainian",         short: "Привіт, як справи?",               keywords: ["привіт", "справи"] },
  vi: { name: "Vietnamese",        short: "Xin chào, bạn khỏe không?",       keywords: ["xin chào", "khỏe"] },
  zh: { name: "Chinese",           short: "你好，你好吗？",                       keywords: ["你好"] },
};

// ---- Pair list (one entry per registry pair constant) ----------------------

export interface NmtPair {
  /** Resource key registered via resources.define() in the consumer. */
  resourceKey: string;
  /** Name of the registry constant exported from @qvac/sdk. */
  constantName: string;
  engine: "Bergamot" | "IndicTrans";
  /** Engine-native source code passed to NmtConfig.from. */
  from: string;
  /** Engine-native target code passed to NmtConfig.to. */
  to: string;
  /** ISO 639-1 source code, used to look up LangEntry. */
  srcIso: string;
  /** ISO 639-1 target code. */
  tgtIso: string;
  /** True when the source side is English (en→XX). */
  fromEn: boolean;
  /** Non-English half of the pair — looked up in LANGUAGES. */
  otherLang: string;
}

// Mirrors the BERGAMOT_EN_<XX> constants present in
// packages/sdk/models/registry/models.ts (101 en-centric pairs total).
const BERGAMOT_EN_TO = [
  "ar","az","bg","bn","bs","ca","cs","da","de","el","es","et","fa","fi","fr",
  "gu","he","hi","hr","hu","id","is","it","ja","kn","ko","lt","lv","ml","ms",
  "nb","nl","no","pl","pt","ro","ru","sk","sl","sq","sr","sv","ta","te","th",
  "tr","uk","vi","zh",
] as const;

const BERGAMOT_FROM_EN = [
  "ar","az","be","bg","bn","bs","ca","cs","da","de","el","es","et","fa","fi",
  "fr","gu","he","hi","hr","hu","id","is","it","ja","kn","ko","lt","lv","ml",
  "ms","mt","nb","nl","nn","no","pl","pt","ro","ru","sk","sl","sq","sr","sv",
  "ta","te","th","tr","uk","vi","zh",
] as const;

function bergamotPair(src: string, tgt: string): NmtPair {
  const fromEn = src === "en";
  return {
    resourceKey: `nmt-pair-bergamot-${src}-${tgt}`,
    constantName: `BERGAMOT_${src.toUpperCase()}_${tgt.toUpperCase()}`,
    engine: "Bergamot",
    from: src,
    to: tgt,
    srcIso: src,
    tgtIso: tgt,
    fromEn,
    otherLang: fromEn ? tgt : src,
  };
}

function indicPair(direction: "en-hi" | "hi-en"): NmtPair {
  const [srcIso, tgtIso] = direction.split("-") as [string, string];
  const fromEn = srcIso === "en";
  return {
    resourceKey: `nmt-pair-indictrans-${direction}`,
    constantName: fromEn
      ? "MARIAN_EN_HI_INDIC_200M_Q4_0"
      : "MARIAN_HI_EN_INDIC_200M_Q4_0",
    engine: "IndicTrans",
    from: fromEn ? "eng_Latn" : "hin_Deva",
    to: fromEn ? "hin_Deva" : "eng_Latn",
    srcIso,
    tgtIso,
    fromEn,
    otherLang: "hi",
  };
}

export const NMT_PAIRS: readonly NmtPair[] = [
  ...BERGAMOT_EN_TO.map((tgt) => bergamotPair("en", tgt)),
  ...BERGAMOT_FROM_EN.map((src) => bergamotPair(src, "en")),
  indicPair("en-hi"),
  indicPair("hi-en"),
];

// ---- TestDefinition generator -----------------------------------------------

const CATEGORY = "translation-nmt-pair";
const SUITE = "nmt-langpairs";
// Bergamot is small (~30MB) and fast; IndicTrans 200M is heavier but still
// fits in a single short-sentence load + decode on desktop. Matches the
// existing translation-bergamot test budget.
const ESTIMATED_MS = 15000;

// Desktop-only. The ~100 bergamot blobs + IndicTrans are too heavy to
// pull reliably over Device Farm's mobile network within the per-test
// budget, and accuracy is platform-independent so a desktop pass is
// sufficient coverage.
const SKIP_MOBILE = {
  reason: "NMT per-pair sanity is desktop-only: model downloads are too heavy for mobile and accuracy is platform-independent",
  platforms: ["mobile-ios", "mobile-android"],
};

function buildSanityTest(pair: NmtPair): TestDefinition | null {
  const lang = LANGUAGES[pair.otherLang];
  if (!lang) return null;
  const text = pair.fromEn ? EN_SHORT : lang.short;
  const contains = pair.fromEn ? lang.keywords : EN_KEYWORDS;
  const expectation: Expectation = { validation: "contains-any", contains };
  return {
    testId: `${CATEGORY}-${pair.engine.toLowerCase()}-${pair.srcIso}-${pair.tgtIso}-basic`,
    params: { text, resource: pair.resourceKey },
    expectation,
    suites: [SUITE],
    skip: SKIP_MOBILE,
    metadata: {
      category: CATEGORY,
      dependency: pair.resourceKey,
      estimatedDurationMs: ESTIMATED_MS,
    },
  };
}

export const translationNmtPairsTests: TestDefinition[] = NMT_PAIRS
  .map(buildSanityTest)
  .filter((t): t is TestDefinition => t !== null);
