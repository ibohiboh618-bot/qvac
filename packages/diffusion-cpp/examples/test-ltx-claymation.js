'use strict'

// ---------------------------------------------------------------------------
// Minimal LTX-2.3 smoke test: "a claymation cat playing piano", 1 step.
//
// This is NOT meant to produce a good clip -- 1 step + tiny resolution just
// exercises the full load -> generate -> mux -> save path as fast as possible.
// For a real clip use examples/generate-video-ltx.js (20 steps, 768x512).
//
// Run:  bare examples/test-ltx-claymation.js
//   or: npm run test:ltx
// Override anything via env, e.g.  STEPS=4 FRAMES=25 npm run test:ltx
// ---------------------------------------------------------------------------

const path = require('bare-path')
const process = require('bare-process')
const fs = require('bare-fs')
const VideoStableDiffusion = require('../video')

const MODELS_DIR = path.resolve(__dirname, '../models')
const OUTPUT_DIR = path.resolve(__dirname, '../output')

// Same distilled weights as generate-video-ltx.js (scripts/download-model-ltx.sh).
const DIFFUSION_MODEL = process.env.LTX_MODEL || 'LTX-2.3-22B-distilled-1.1-Q8_0.gguf'
const LLM_MODEL = process.env.LTX_LLM || 'gemma-3-12b-it-UD-Q4_K_XL.gguf'
const VIDEO_VAE = process.env.LTX_VAE || 'ltx-2.3-22b-distilled_video_vae.safetensors'
const AUDIO_VAE = process.env.LTX_AUDIO_VAE || 'ltx-2.3-22b-distilled_audio_vae.safetensors'
const EMBEDDINGS_CONNECTORS =
  process.env.LTX_CONNECTORS || 'ltx-2.3-22b-distilled_embeddings_connectors.safetensors'

// Minimal-but-valid LTX shape: dims must be x32, frames must be (8*k + 1).
const PROMPT = process.env.PROMPT || 'a claymation cat playing piano'
const NEG_PROMPT = process.env.NEG_PROMPT || 'blurry, low quality, watermark'
const WIDTH = parseInt(process.env.WIDTH || '512', 10) // 512 = 16 x 32
const HEIGHT = parseInt(process.env.HEIGHT || '320', 10) // 320 = 10 x 32
const VIDEO_FRAMES = parseInt(process.env.FRAMES || '9', 10) // 8*1 + 1 (smallest)
const FPS = parseInt(process.env.FPS || '24', 10)
const STEPS = parseInt(process.env.STEPS || '1', 10)
const CFG_SCALE = parseFloat(process.env.CFG_SCALE || '1.0')
const SEED = parseInt(process.env.SEED || '42', 10)
const TEMPORAL_TILING = (process.env.TEMPORAL_TILING || 'true') === 'true'

async function main () {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true })

  if (WIDTH % 32 !== 0 || HEIGHT % 32 !== 0) {
    console.error(`WIDTH/HEIGHT must be multiples of 32 (got ${WIDTH}x${HEIGHT}).`)
    process.exit(1)
  }
  if (VIDEO_FRAMES < 9 || (VIDEO_FRAMES - 1) % 8 !== 0 || VIDEO_FRAMES > 257) {
    console.error(`FRAMES must be (8*k + 1) in [9, 257] (got ${VIDEO_FRAMES}).`)
    process.exit(1)
  }

  console.log('LTX-2.3 smoke test (1 step)')
  console.log('===========================')
  console.log('Prompt :', PROMPT)
  console.log('Size   :', `${WIDTH}x${HEIGHT}`)
  console.log('Frames :', VIDEO_FRAMES, `(@${FPS} fps)`)
  console.log('Steps  :', STEPS)
  console.log('Seed   :', SEED)
  console.log()

  const model = new VideoStableDiffusion({
    files: {
      model: path.join(MODELS_DIR, DIFFUSION_MODEL),
      llm: path.join(MODELS_DIR, LLM_MODEL),
      vae: path.join(MODELS_DIR, VIDEO_VAE),
      audioVae: path.join(MODELS_DIR, AUDIO_VAE),
      embeddingsConnectors: path.join(MODELS_DIR, EMBEDDINGS_CONNECTORS)
    },
    config: {
      threads: 4,
      device: 'gpu',
      diffusion_fa: true,
      offload_to_cpu: true,
      vae_tiling: true,
      vae_conv_direct: true
    },
    opts: { stats: true },
    logger: console
  })

  try {
    console.log('Loading weights...')
    const tLoad = Date.now()
    await model.load()
    console.log(`Loaded in ${((Date.now() - tLoad) / 1000).toFixed(1)}s\n`)

    console.log('Generating...')
    const tGen = Date.now()

    const response = await model.run({
      mode: 'txt2vid',
      prompt: PROMPT,
      negative_prompt: NEG_PROMPT,
      width: WIDTH,
      height: HEIGHT,
      video_frames: VIDEO_FRAMES,
      fps: FPS,
      steps: STEPS,
      cfg_scale: CFG_SCALE,
      temporal_tiling: TEMPORAL_TILING,
      seed: SEED
    })

    let avi = null
    let stats = null
    response.on('stats', (s) => { stats = s })

    await response
      .onUpdate((data) => {
        if (data instanceof Uint8Array) {
          avi = data
        } else if (typeof data === 'string') {
          try {
            const tick = JSON.parse(data)
            if ('step' in tick && 'total' in tick) {
              process.stdout.write(`\r  step ${tick.step}/${tick.total}  `)
            }
          } catch (_) {}
        }
      })
      .await()

    process.stdout.write('\n')
    console.log(`Generated in ${((Date.now() - tGen) / 1000).toFixed(1)}s`)

    if (stats) {
      const hasAudio = stats.hasAudio === 1 || stats.hasAudio === true
      console.log('Audio  :', hasAudio ? `yes (${stats.audioSampleRate} Hz)` : 'none')
    }

    if (avi) {
      const outPath = path.join(OUTPUT_DIR, `ltx_test_claymation_seed${SEED}.avi`)
      fs.writeFileSync(outPath, avi)
      console.log(`Saved -> ${outPath} (${avi.length.toLocaleString()} bytes)`)
      console.log('Tip: play in VLC to hear the muxed audio track.')
    } else {
      console.warn('No AVI buffer received -- check native logs above.')
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
