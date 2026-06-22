'use strict'

const path = require('bare-path')
const process = require('bare-process')
const fs = require('bare-fs')
const ImgStableDiffusion = require('../index')

// ---------------------------------------------------------------------------
// Model files — downloaded via: ./scripts/download-model-ideogram.sh
// ---------------------------------------------------------------------------
const MODELS_DIR = path.resolve(__dirname, '../models')
const OUTPUT_DIR = path.resolve(__dirname, '../output')

const MODEL_NAME = 'ideogram4-Q4_0.gguf' // conditional diffusion model
const UNCOND_MODEL = 'ideogram4_uncond-Q4_0.gguf' // unconditional model (CFG)
const LLM_MODEL = 'Qwen3-VL-8B-Instruct-Q4_K_M.gguf' // text encoder
const VAE_MODEL = 'flux2-vae.safetensors' // FLUX.2-family VAE (shared)

// ---------------------------------------------------------------------------
// Generation params — edit freely
// ---------------------------------------------------------------------------
// Ideogram 4 expects a *structured JSON caption* (not a plain sentence). The
// schema below mirrors leejet's reference prompt (docs/ideogram4.md):
//   - high_level_description : one-paragraph summary of the whole image
//   - style_description      : aesthetics / lighting / photo / medium / palette
//   - compositional_deconstruction : canvas + background + layout + an ordered
//     `elements` array of { type: "obj" | "text", desc }
// The addon forwards `prompt` straight through to the model verbatim, so we
// JSON.stringify the structure below. Feeding a plain sentence (or an off-schema
// object) yields degenerate conditioning and Ideogram falls back to rendering
// its built-in "Image blocked by safety filter" placeholder.
const PROMPT = JSON.stringify({
  high_level_description:
    'A square 1024 x 1024 luxury fashion magazine cover featuring exactly one short chubby fluffy cat as the main model. The cat sits on a soft ivory studio floor, facing the viewer with a stylish calm expression, wearing tiny black sunglasses, a red silk scarf, and a small gold collar charm. In front of the cat on the floor is a wide horizontal luxury nameplate that clearly reads ideogram4.cpp. The whole design feels premium, fashionable, clean, and editorial.',
  style_description: {
    aesthetics:
      'luxury fashion magazine cover, high-end pet couture campaign, minimalist editorial design, elegant studio photography, soft paper texture, refined typography, fashionable and polished',
    lighting:
      'Soft diffused studio lighting, gentle spotlight on the cat, subtle floor shadow, warm ivory highlights, clean separation between subject and background',
    photo:
      'high-resolution fashion editorial photography look, front-facing cat portrait, crisp fur details, glossy sunglasses, clear readable nameplate text, shallow depth of field',
    medium: 'mixed media fashion photography and premium editorial graphic design',
    color_palette: ['#F4EFE7', '#111111', '#D8B56D', '#B73A3A', '#FFFFFF', '#8A7A6A']
  },
  compositional_deconstruction: {
    canvas:
      'Square 1024 x 1024 canvas with a normal upright orientation. Do not rotate the poster or any text. Use a clean fashion magazine cover layout.',
    background:
      'Warm ivory studio backdrop with subtle paper grain, a soft spotlight gradient, faint floor shadow, and a few minimal gold editorial lines. The background is spacious, premium, and uncluttered.',
    layout:
      'Top center has a small elegant headline. Center area features one cat as the main fashion model. Lower foreground has a wide horizontal luxury nameplate placed on the floor in front of the cat. Bottom center has a small footer. All text is horizontal, upright, and readable left to right.',
    elements: [
      { type: 'text', desc: 'Top center headline reading LOOK WHAT I FOUND in a refined high-fashion serif font. The headline is horizontal, centered, elegant, and secondary to the nameplate text.' },
      { type: 'obj', desc: 'Exactly one short chubby fluffy cat sitting in the center like a luxury fashion model. The cat has a large round head, compact body, short legs, soft detailed fur, expressive eyes, and a calm confident pose. The cat is cute and rounded, not tall, not stretched, not duplicated.' },
      { type: 'obj', desc: 'Tiny glossy black sunglasses worn naturally by the cat, slightly oversized but still showing the cat face clearly. The sunglasses add a chic fashion-editorial attitude.' },
      { type: 'obj', desc: 'A red silk scarf tied neatly around the cat neck, with soft folds and a couture feeling. The scarf must not cover the cat face or the nameplate.' },
      { type: 'obj', desc: 'A small gold collar charm or fashion accessory under the scarf, subtle and premium, adding a luxury campaign detail.' },
      { type: 'obj', desc: 'In the lower foreground, place a wide horizontal luxury nameplate on the floor in front of the cat. The nameplate is low, flat, landscape-oriented, much wider than tall, like a fashion show seat card or premium display plaque. It is centered, front-facing, level, and fully visible. It must not become vertical, tall, standing, rotated, or side-facing.' },
      { type: 'text', desc: 'Print the exact text ideogram4.cpp only on the wide horizontal nameplate. Use clean bold black lettering, perfectly spelled, lowercase, with the number 4 and .cpp extension. The text must fit completely inside the nameplate, stay horizontal, and be readable from left to right.' },
      { type: 'obj', desc: 'Add sparse premium editorial accents around the edges: thin gold lines, small code brackets, tiny cursor marks, subtle dots, and minimal geometric details. No extra cats, no stickers, no animal faces, no busy decorations.' },
      { type: 'text', desc: 'Bottom center footer reading tiny paws, big compile energy in a small refined monospace or editorial font. The footer is horizontal, centered, understated, and much smaller than the nameplate text.' }
    ]
  }
})

const STEPS = 20
const WIDTH = 1024
const HEIGHT = 1024
const CFG_SCALE = 7.0 // Ideogram 4 uses *real* CFG (via the uncond model); 7.0 = lib default
const SEED = 42 // -1 = random

async function main () {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true })

  console.log('Ideogram 4 — text-to-image inference')
  console.log('====================================')
  console.log('Prompt :', PROMPT)
  console.log('Steps  :', STEPS)
  console.log('Size   :', `${WIDTH}x${HEIGHT}`)
  console.log('Seed   :', SEED)
  console.log()

  const model = new ImgStableDiffusion({
    files: {
      model: path.join(MODELS_DIR, MODEL_NAME),
      uncondModel: path.join(MODELS_DIR, UNCOND_MODEL),
      llm: path.join(MODELS_DIR, LLM_MODEL),
      vae: path.join(MODELS_DIR, VAE_MODEL)
    },
    config: {
      threads: 4,
      diffusion_fa: true,
      offload_to_cpu: true
    },
    logger: console
  })

  try {
    // ── 1. Load weights ───────────────────────────────────────────────────────
    console.log('Loading model weights...')
    const tLoad = Date.now()
    await model.load()
    console.log(`Loaded in ${((Date.now() - tLoad) / 1000).toFixed(1)}s\n`)

    // ── 2. Start generation ───────────────────────────────────────────────────
    console.log('Starting generation...')
    const tGen = Date.now()

    const response = await model.run({
      prompt: PROMPT,
      steps: STEPS,
      width: WIDTH,
      height: HEIGHT,
      cfg_scale: CFG_SCALE,
      seed: SEED
    })

    // ── 3. Stream progress + collect image bytes ──────────────────────────────
    const images = []

    await response
      .onUpdate((data) => {
        if (data instanceof Uint8Array) {
          // PNG-encoded output image
          images.push(data)
        } else if (typeof data === 'string') {
          try {
            const tick = JSON.parse(data)
            if ('step' in tick && 'total' in tick) {
              const pct = Math.round((tick.step / tick.total) * 100)
              const bar = '█'.repeat(Math.floor(pct / 5)).padEnd(20, '░')
              process.stdout.write(`\r  [${bar}] ${tick.step}/${tick.total} steps`)
            }
          } catch (_) {}
        }
      })
      .await()

    process.stdout.write('\n')
    console.log(`\nGenerated in ${((Date.now() - tGen) / 1000).toFixed(1)}s`)
    console.log(`Got ${images.length} image(s)`)

    // ── 4. Save each image to disk ────────────────────────────────────────────
    for (let i = 0; i < images.length; i++) {
      const outPath = path.join(OUTPUT_DIR, `ideogram_seed${SEED}_${i}.png`)
      fs.writeFileSync(outPath, images[i])
      console.log(`Saved → ${outPath}`)
    }
  } finally {
    console.log('\nUnloading model...')
    await model.unload()
    console.log('Done.')
  }
}

main().catch(err => {
  console.error('Fatal:', err.message || err)
  process.exit(1)
})
