import { loadModel, unloadModel, diffusion } from "@qvac/sdk";
import fs from "fs";

// Minimal diffusion example — single GGUF model, no companion files needed.
// Works with SD 1.x / 2.x all-in-one models.
const modelSrc = process.argv[2] || "/models/sd-v1-4-Q8_0.gguf";
const prompt = process.argv[3] || "a photo of a cat sitting on a windowsill";

const modelId = await loadModel({
  modelSrc,
  modelType: "diffusion",
});

const { outputs } = diffusion({ modelId, prompt });
const buffers = await outputs;

fs.writeFileSync("output.png", buffers[0]!);
console.log("Saved: output.png");

await unloadModel({ modelId, clearStorage: false });
process.exit(0);
