'use strict'

const test = require('brittle')
const {
  splitText,
  splitBySentence,
  splitByParagraphs,
  mergeShortChunks,
  splitLongChunk,
  splitOnWordBoundary,
  MAX_CHUNK_GRAPHEMES
} = require('../../lib/text-splitter')

test('splitBySentence: splits on period', (t) => {
  const result = splitBySentence('Hello world. Goodbye world.')
  t.alike(result, ['Hello world.', 'Goodbye world.'])
})

test('splitBySentence: splits on exclamation and question marks', (t) => {
  const result = splitBySentence('Hello! How are you? Fine.')
  t.alike(result, ['Hello!', 'How are you?', 'Fine.'])
})

test('splitBySentence: handles trailing text without terminator', (t) => {
  const result = splitBySentence('Hello world. No ending')
  t.alike(result, ['Hello world.', 'No ending'])
})

test('splitBySentence: handles CJK terminators', (t) => {
  const result = splitBySentence('Hello。World！Done？')
  t.alike(result, ['Hello。', 'World！', 'Done？'])
})

test('splitBySentence: returns whole text when no terminators', (t) => {
  const result = splitBySentence('No terminators here')
  t.alike(result, ['No terminators here'])
})

test('splitByParagraphs: splits on double newlines', (t) => {
  const result = splitByParagraphs('First paragraph.\n\nSecond paragraph.')
  t.alike(result, ['First paragraph.', 'Second paragraph.'])
})

test('splitByParagraphs: trims whitespace', (t) => {
  const result = splitByParagraphs('  First.  \n\n  Second.  ')
  t.alike(result, ['First.', 'Second.'])
})

test('splitByParagraphs: filters empty paragraphs', (t) => {
  const result = splitByParagraphs('First.\n\n\n\nSecond.')
  t.alike(result, ['First.', 'Second.'])
})

test('mergeShortChunks: merges chunks shorter than MIN_CHUNK_GRAPHEMES', (t) => {
  const result = mergeShortChunks(['Hello there.', 'Bye.', 'Last sentence.'])
  t.alike(result, ['Hello there.', 'Bye. Last sentence.'])
})

test('mergeShortChunks: keeps chunks at or above MIN_CHUNK_GRAPHEMES', (t) => {
  const result = mergeShortChunks(['First sentence here.', 'Second sentence here.'])
  t.alike(result, ['First sentence here.', 'Second sentence here.'])
})

test('mergeShortChunks: handles empty input', (t) => {
  const result = mergeShortChunks([])
  t.alike(result, [])
})

test('splitLongChunk: returns short text unchanged', (t) => {
  const result = splitLongChunk('Short text.')
  t.alike(result, ['Short text.'])
})

test('splitLongChunk: splits on commas when over MAX_CHUNK_GRAPHEMES', (t) => {
  const clause = 'word '.repeat(30).trim()
  const longText = clause + ', ' + clause + ', ' + clause
  const result = splitLongChunk(longText)
  t.ok(result.length > 1, 'Should split into multiple parts')
  t.ok(result.every(c => [...c].length <= MAX_CHUNK_GRAPHEMES), 'Every chunk should be within MAX_CHUNK_GRAPHEMES')
})

test('splitLongChunk: falls back to word boundary when no secondary delimiters', (t) => {
  const longText = 'word '.repeat(100).trim()
  t.ok([...longText].length > MAX_CHUNK_GRAPHEMES, 'Input should exceed MAX_CHUNK_GRAPHEMES')
  const result = splitLongChunk(longText)
  t.ok(result.length > 1, 'Should split into multiple parts')
  t.ok(result.every(c => [...c].length <= MAX_CHUNK_GRAPHEMES), 'Every chunk should be within MAX_CHUNK_GRAPHEMES')
})

test('splitOnWordBoundary: splits long text at word boundaries', (t) => {
  const longText = 'word '.repeat(100).trim()
  const result = splitOnWordBoundary(longText, 50)
  t.ok(result.length > 1, 'Should split into multiple parts')
  t.ok(result.every(c => [...c].length <= 50), 'Every chunk should respect the max')
})

test('splitText: handles single sentence', (t) => {
  const result = splitText('Hello world.')
  t.alike(result, ['Hello world.'])
})

test('splitText: splits multiple sentences into chunks', (t) => {
  const result = splitText('This is the first sentence. This is the second sentence. And a third one.')
  t.is(result.length, 3)
  t.ok(result.every(c => c.length > 0))
})

test('splitText: handles paragraphs with sentences', (t) => {
  const result = splitText('First paragraph sentence one. Sentence two.\n\nSecond paragraph sentence.')
  t.ok(result.length >= 2)
  t.ok(result.includes('Second paragraph sentence.'))
})

test('splitText: returns empty array for empty text', (t) => {
  const result = splitText('')
  t.alike(result, [])
})

test('splitText: returns trimmed text when no sentence terminators', (t) => {
  const result = splitText('just some text without terminators')
  t.alike(result, ['just some text without terminators'])
})

test('splitText: merges short fragments together', (t) => {
  const result = splitText('Hi. Ok. This is a much longer sentence that stands alone.')
  t.ok(result.length <= 2, 'Short fragments should be merged')
  t.ok(result[0].includes('Hi.'))
  t.ok(result[0].includes('Ok.'))
})

test('splitText: handles whitespace-only input', (t) => {
  const result = splitText('   \n\n   ')
  t.alike(result, [])
})

test('splitText: enforces MAX_CHUNK_GRAPHEMES on long sentences', (t) => {
  const longSentence = 'word '.repeat(100).trim() + '.'
  t.ok([...longSentence].length > MAX_CHUNK_GRAPHEMES, 'Input sentence should exceed MAX_CHUNK_GRAPHEMES')
  const result = splitText(longSentence)
  t.ok(result.length > 1, 'Should split long sentence into multiple chunks')
  t.ok(result.every(c => [...c].length <= MAX_CHUNK_GRAPHEMES), 'Every chunk should be within MAX_CHUNK_GRAPHEMES')
})

test('splitText: enforces MAX_CHUNK_GRAPHEMES on text without terminators', (t) => {
  const longText = 'word '.repeat(100).trim()
  const result = splitText(longText)
  t.ok(result.length > 1, 'Should split long text without terminators')
  t.ok(result.every(c => [...c].length <= MAX_CHUNK_GRAPHEMES), 'Every chunk should be within MAX_CHUNK_GRAPHEMES')
})

test('splitText: enforces MAX_CHUNK_GRAPHEMES on long paragraph without sentence terminators', (t) => {
  const longParagraph = 'this is a clause with commas, '.repeat(20).trim()
  t.ok([...longParagraph].length > MAX_CHUNK_GRAPHEMES, 'Input should exceed MAX_CHUNK_GRAPHEMES')
  const result = splitText(longParagraph)
  t.ok(result.length > 1, 'Should split long paragraph')
  t.ok(result.every(c => [...c].length <= MAX_CHUNK_GRAPHEMES), 'Every chunk should be within MAX_CHUNK_GRAPHEMES')
})
