# Unit Test Models

This directory contains NMT models required for C++ unit tests.

## Required Models

| Model | Size | Purpose |
|-------|------|---------|
| `ggml-indictrans2-en-indic-dist-200M-q4_0.bin` | ~122M | English → Indic (IndicTrans tests) |

## Setup Options

### Option 1: Download from HuggingFace (Recommended)

```bash
# Step 1: Download model
mkdir -p models/unit-test
curl -sL "https://huggingface.co/olyas/indictrans2-ggml/resolve/main/ggml-indictrans2-en-indic-dist-200M-q4_0.bin" -o models/unit-test/ggml-indictrans2-en-indic-dist-200M-q4_0.bin

# Step 2: Run tests
./build/addon/tests/addon-test
```

### Option 2: Run Only Tests That Don't Need Models

```bash
# Bergamot validation tests (no models needed)
./build/addon/tests/addon-test --gtest_filter="BergamotValidationTest.*:BergamotBatchTest.*:NmtConfigTest.*"
```

## What Happens If Models Are Missing?

- Tests that require missing models will be **skipped** with `GTEST_SKIP()`
- You'll see messages like: `Model not found: ... See models/unit-test/README.md for setup instructions.`
- Bergamot validation tests will still **run** (they don't need models)

## CI/CD

In CI/CD pipelines, models are automatically downloaded before running tests.
See `.github/workflows/cpp-tests.yaml` for the automated configuration.

## Verifying Setup

```bash
ls -la models/unit-test/
```

You should see `.bin` files (or symlinks pointing to `../model.bin` etc.).
