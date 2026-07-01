// QVAC-20557 R2 probe: validate that setting TTS_CPP_GPU_DUMP_DIR via JS
// proc.env (the exact gpu-smoke.test.js mechanism) reaches the native
// getenv("TTS_CPP_GPU_DUMP_DIR") in toEngineOptions, so per-stage raw f32 dumps
// land on disk. Loads Supertonic on GPU, synthesizes once, then lists the dump
// dir and writes the result to TTS_OUT.
const fs = require('bare-fs')
const path = require('bare-path')
const proc = require('bare-process')
const TTSGgml = require('./index')

const HERE = __dirname
const SUPERTONIC = path.join(HERE, 'models', 'supertonic2.gguf')
const OUT = (proc.env && proc.env.TTS_OUT) || path.join(HERE, 'dump-probe.out')
const DUMP = path.join(HERE, 'gpu-diag')

function writeOut (obj) {
  try { fs.writeFileSync(OUT, JSON.stringify(obj) + '\n') } catch (_e) {}
  console.log('DUMP_PROBE ' + JSON.stringify(obj))
}

async function main () {
  try { fs.mkdirSync(DUMP, { recursive: true }) } catch (_e) { try { fs.mkdirSync(DUMP) } catch (_e2) {} }
  // The hop under test: JS proc.env assignment must reach native getenv.
  proc.env.TTS_CPP_GPU_DUMP_DIR = DUMP

  const model = new TTSGgml({
    engine: TTSGgml.ENGINE_SUPERTONIC,
    files: { supertonicModel: SUPERTONIC },
    voice: 'F1',
    config: { language: 'en', useGPU: true },
    opts: { stats: true }
  })
  await model.load()
  const response = await model.run({ input: 'GPU dump probe.', type: 'text' })
  await response.onUpdate(() => {}).await()
  try { await model.unload() } catch (_e) {}

  let files = []
  try { files = fs.readdirSync(DUMP) } catch (_e) {}
  writeOut({ dumpDir: DUMP, fileCount: files.length, files })
}

main().catch(e => writeOut({ error: String(e && e.stack ? e.stack : e) }))
