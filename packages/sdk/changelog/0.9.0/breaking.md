# 💥 Breaking Changes v0.9.0

## Deprecate Opus NMT Engine

PR: [#1214](https://github.com/tetherto/qvac/pull/1214)

Opus is no longer a supported NMT engine in the SDK. All `MARIAN_OPUS_*` model constants, the `"Opus"` engine literal, `opusConfigSchema`, `MARIAN_LANGUAGES`, `MarianLanguage`, and `generateNmtOpusName` have been removed. Use Bergamot for European language pairs and IndicTrans for Indic language pairs.

**BEFORE:**

```typescript
import { MARIAN_OPUS_EN_IT_Q4_0 } from "@tetherto/qvac-sdk";

const modelId = await loadModel({
  modelSrc: MARIAN_OPUS_EN_IT_Q4_0,
  modelType: "nmt",
  modelConfig: {
    engine: "Opus",
    from: "en",
    to: "it",
  },
});
```

**AFTER:**

```typescript
import { BERGAMOT_EN_ES } from "@tetherto/qvac-sdk";

const modelId = await loadModel({
  modelSrc: BERGAMOT_EN_ES,
  modelType: "nmt",
  modelConfig: {
    engine: "Bergamot",
    from: "en",
    to: "es",
  },
});
```

### Migration Guide

| Previously used | Replace with |
|---|---|
| `engine: "Opus"` | `engine: "Bergamot"` |
| `MARIAN_OPUS_EN_*` / `MARIAN_OPUS_*_EN` | Corresponding `BERGAMOT_EN_*` / `BERGAMOT_*_EN` constant |
| `MARIAN_LANGUAGES` | `BERGAMOT_LANGUAGES` |
| `MarianLanguage` | `BergamotLanguage` |
| Direct non-English pairs (e.g., de-fr) | Use Bergamot pivot translation via `pivotModel` config |

---
