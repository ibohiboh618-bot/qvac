'use strict'

// LTX-2.3 text-to-video end-to-end smoke test.
//
// Minimal smoke test: "a claymation cat playing piano", 1 step, 9 frames, 512x320.
// This is NOT meant to produce a good clip -- 1 step + tiny resolution just
// exercises the full load -> generate -> audio-mux -> save path as fast as possible.
//
// The LTX-2 ops (temporal_tiling VAE, audio VAE, embeddings connectors) require the
// ggml fork and stable-diffusion-cpp fork pinned through the qvac vcpkg registry.

const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')
const proc = require('bare-process')
const test = require('brittle')
const VideoStableDiffusion = require('@qvac/diffusion-cpp/video')
const { setupJsLogger, ensureModelPath } = require('./utils')

const isMobile = os.platform() === 'ios' || os.platform() === 'android'
const noGpu = proc.env && proc.env.NO_GPU === 'true'
// Skip LTX tests on mobile and on CPU-only runners (NO_GPU).
// LTX also runs on darwin (unlike Wan which OOMs), but you can skip via env if needed.
const skip = isMobile || noGpu

// Log skip status for CI visibility
console.log('[LTX Video Tests] Platform:', os.platform(), 'Arch:', os.arch(), 'NO_GPU:', noGpu, '→ Skip:', skip)

const LTX_MODELS_DIR = proc.env.LTX_MODELS_DIR || path.join(__dirname, '../../models')

// LTX-2.3 distilled model files
const LTX_FILES = [
  {
    key: 'model',
    name: 'LTX-2.3-22B-distilled-1.1-Q8_0.gguf',
    url: 'https://huggingface.co/Lightricks/LTX-Video/resolve/main/LTX-2.3-22B-distilled-1.1-Q8_0.gguf'
  },
  {
    key: 'llm',
    name: 'gemma-3-12b-it-UD-Q4_K_XL.gguf',
    url: 'https://huggingface.co/Lightricks/LTX-Video/resolve/main/gemma-3-12b-it-UD-Q4_K_XL.gguf'
  },
  {
    key: 'vae',
    name: 'ltx-2.3-22b-distilled_video_vae.safetensors',
    url: 'https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-2.3-22b-distilled_video_vae.safetensors'
  },
  {
    key: 'audioVae',
    name: 'ltx-2.3-22b-distilled_audio_vae.safetensors',
    url: 'https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-2.3-22b-distilled_audio_vae.safetensors'
  },
  {
    key: 'embeddingsConnectors',
    name: 'ltx-2.3-22b-distilled_embeddings_connectors.safetensors',
    url: 'https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltx-2.3-22b-distilled_embeddings_connectors.safetensors'
  }
]

const MODELS_DIR = LTX_MODELS_DIR
const OUTPUT_DIR = path.resolve(__dirname, '../../output')

// Minimal-but-valid LTX shape: dims must be x32, frames must be (8*k + 1).
const PROMPT = proc.env.PROMPT || 'a claymation cat playing piano'
const NEG_PROMPT = proc.env.NEG_PROMPT || 'blurry, low quality, watermark'
const WIDTH = parseInt(proc.env.WIDTH || '512', 10) // 512 = 16 x 32
const HEIGHT = parseInt(proc.env.HEIGHT || '320', 10) // 320 = 10 x 32
const VIDEO_FRAMES = parseInt(proc.env.FRAMES || '9', 10) // 8*1 + 1 (smallest)
const FPS = parseInt(proc.env.FPS || '24', 10)
const STEPS = parseInt(proc.env.STEPS || '1', 10)
const CFG_SCALE = parseFloat(proc.env.CFG_SCALE || '1.0')
const SEED = parseInt(proc.env.SEED || '42', 10)
const TEMPORAL_TILING = (proc.env.TEMPORAL_TILING || 'true') === 'true'
const DEVICE = proc.env.LTX_DEVICE || 'gpu'

setupJsLogger()

test('LTX-2.3 smoke test (T2V, 1 step, 9 frames)', { skip }, async (t) => {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true })

  if (WIDTH % 32 !== 0 || HEIGHT % 32 !== 0) {
    t.fail(`WIDTH/HEIGHT must be multiples of 32 (got ${WIDTH}x${HEIGHT})`)
    return
  }
  if (VIDEO_FRAMES < 9 || (VIDEO_FRAMES - 1) % 8 !== 0 || VIDEO_FRAMES > 257) {
    t.fail(`FRAMES must be (8*k + 1) in [9, 257] (got ${VIDEO_FRAMES})`)
    return
  }

  // Ensure all model files exist
  const files = {}
  for (const model of LTX_FILES) {
    const modelPath = await ensureModelPath(model.name, model.url, MODELS_DIR)
    files[model.key] = modelPath
  }

  const model = new VideoStableDiffusion({
    files,
    config: {
      threads: 4,
      device: DEVICE,
      diffusion_fa: true,
      offload_to_cpu: true,
      vae_tiling: true,
      vae_conv_direct: true
    },
    opts: { stats: true },
    logger: console
  })

  try {
    console.log('\n[LTX] Loading weights...')
    const tLoad = Date.now()
    await model.load()
    console.log(`[LTX] Loaded in ${((Date.now() - tLoad) / 1000).toFixed(1)}s`)

    console.log('[LTX] Generating...')
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
              process.stdout.write(`\r[LTX]   step ${tick.step}/${tick.total}  `)
            }
          } catch (_) {}
        }
      })
      .await()

    process.stdout.write('\n')
    console.log(`[LTX] Generated in ${((Date.now() - tGen) / 1000).toFixed(1)}s`)

    if (stats) {
      const hasAudio = stats.hasAudio === 1 || stats.hasAudio === true
      console.log('[LTX] Audio  :', hasAudio ? `yes (${stats.audioSampleRate} Hz)` : 'none')
    }

    // Verify output
    t.ok(avi, 'AVI buffer generated')
    t.ok(avi.length > 0, 'AVI buffer has content')
    t.ok(stats, 'Stats returned')

    if (stats && (stats.hasAudio === 1 || stats.hasAudio === true)) {
      t.ok(stats.audioSampleRate > 0, 'Audio sample rate is positive')
    }

    if (avi) {
      const outPath = path.join(OUTPUT_DIR, `ltx_t2v_test_seed${SEED}.avi`)
      fs.writeFileSync(outPath, avi)
      console.log(`[LTX] Saved -> ${outPath} (${avi.length.toLocaleString()} bytes)`)
      console.log('[LTX] Tip: play in VLC to hear the muxed audio track.')
    }
  } finally {
    console.log('[LTX] Unloading model...')
    await model.unload()
    console.log('[LTX] Done.')
  }
})
