/* eslint-disable */
'use strict'
// QVAC-21320: two-image VLM benchmark fixture (elephant + fruit-plate) — addon test
// assets from packages/llm-llamacpp/media/, staged by stage.cjs (NOT the lmms-eval
// fixture store). One synthetic task 'describe'; gold:[] because these are free-form
// "describe the image" prompts with no lmms-eval gold — results are based on
// mmproj-encode CPU-vs-GPU speed + CPU/GPU output neutrality (keyword presence), not %.
module.exports = {
  "tasks": [
    "describe"
  ],
  "samplesPerTask": 2,
  "items": [
    {
      "id": "elephant",
      "task": "describe",
      "metric": "vqa",
      "prompt": "Describe the image briefly in one sentence.",
      "gold": [],
      "image": "elephant.jpg",
      "width": 612,
      "height": 408,
      "license": "addon test asset (media/elephant.jpg)"
    },
    {
      "id": "fruit-plate",
      "task": "describe",
      "metric": "vqa",
      "prompt": "Describe the image briefly in one sentence.",
      "gold": [],
      "image": "fruitPlate.png",
      "width": 2250,
      "height": 3000,
      "license": "addon test asset (media/fruitPlate.png)"
    }
  ]
}
