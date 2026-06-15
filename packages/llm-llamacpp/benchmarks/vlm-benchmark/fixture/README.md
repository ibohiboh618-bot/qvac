# VLM benchmark fixture images

The benchmark's test images live here. **The image files are NOT in git** — only this
README is tracked. They are stored in a private S3 bucket and must be present locally
before the harness can run.

## Get the images (local runs)

Copy them from the fixture object store into this folder:

```bash
aws s3 sync s3://tether-ai-dev/vlm-benchmark/ packages/llm-llamacpp/benchmarks/vlm-benchmark/fixture/
```

(CI does this automatically; you only need it for local runs.) The image filenames are
referenced by `../fixture.data.cjs` (e.g. `vlmx-textvqa_0.jpg`).

## Add new images

1. Drop the new `.jpg`/`.png` files here and add their entries (task, prompt, gold,
   width/height, license) to `../fixture.data.cjs`.
2. Upload them to S3 so CI and other devs get them:
   ```bash
   aws s3 sync packages/llm-llamacpp/benchmarks/vlm-benchmark/fixture/ s3://tether-ai-dev/vlm-benchmark/ --exclude README.md
   ```
3. New images can't be exercised in CI until they're on S3 (CI pulls from S3, not git).

`build-fixture.cjs` can regenerate the open-licensed datasets automatically; hand-added
images (e.g. curated OCR samples) are kept as-is.

## Adding OCR tasks (`ocr-line`, `ocr-page`)

The OCR tasks are hand-curated (their source images aren't on the open-license allowlist
the generator enforces). For each OCR image, add an item to `../fixture.data.cjs`:

```js
{
  "id": "ocr-line_0",            // or ocr-page_0
  "task": "ocr-line",            // ocr-line = single word/line · ocr-page = full document
  "metric": "ocr",               // scored by CER/WER/BLEU (separate report table)
  "prompt": "Read the text in the image. Output only the text, nothing else.",
  "gold": ["CENTRE"],            // the exact reference text (page: the full transcription)
  "image": "vlmx-ocrline_0.jpg",
  "width": 320, "height": 96,
  "license": "<source license>"
}
```

Prompt convention: **line** → `"Read the text in the image. Output only the text, nothing else."`;
**page** → `"Transcribe all the text in this document, preserving reading order. Output only the text."`
`gold` is the literal reference string (CER/WER/BLEU compare against it; multiple entries =
acceptable alternative transcriptions). Then upload the images to S3 as above.
