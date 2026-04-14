'use strict'

const SENTENCE_TERMINATORS = /([.!?。！？])\s*/g
const SECONDARY_DELIMITERS = /([,;:，；：])\s*/g

const MIN_CHUNK_GRAPHEMES = 10
const MAX_CHUNK_GRAPHEMES = 250

function splitBySentence (text) {
  const parts = []
  let lastIndex = 0

  text.replace(SENTENCE_TERMINATORS, (match, terminator, offset) => {
    const end = offset + terminator.length
    parts.push(text.slice(lastIndex, end).trim())
    lastIndex = offset + match.length
  })

  const remaining = text.slice(lastIndex).trim()
  if (remaining.length > 0) {
    parts.push(remaining)
  }

  return parts
}

function splitOnDelimiters (text, pattern) {
  const parts = []
  let lastIndex = 0

  text.replace(pattern, (match, delimiter, offset) => {
    const end = offset + delimiter.length
    parts.push(text.slice(lastIndex, end).trim())
    lastIndex = offset + match.length
  })

  const remaining = text.slice(lastIndex).trim()
  if (remaining.length > 0) {
    parts.push(remaining)
  }

  return parts
}

function splitOnWordBoundary (text, maxGraphemes) {
  const words = text.split(/\s+/)
  const parts = []
  let current = ''

  for (const word of words) {
    const candidate = current.length === 0 ? word : current + ' ' + word
    if ([...candidate].length > maxGraphemes && current.length > 0) {
      parts.push(current)
      current = word
    } else {
      current = candidate
    }
  }

  if (current.length > 0) {
    parts.push(current)
  }

  return parts
}

function splitLongChunk (text) {
  if ([...text].length <= MAX_CHUNK_GRAPHEMES) {
    return [text]
  }

  const subParts = splitOnDelimiters(text, SECONDARY_DELIMITERS)
  const result = []

  for (const part of subParts) {
    if ([...part].length <= MAX_CHUNK_GRAPHEMES) {
      result.push(part)
    } else {
      const wordSplit = splitOnWordBoundary(part, MAX_CHUNK_GRAPHEMES)
      for (const ws of wordSplit) {
        result.push(ws)
      }
    }
  }

  return result
}

function mergeShortChunks (chunks) {
  const merged = []
  let buffer = ''

  for (const chunk of chunks) {
    if (buffer.length === 0) {
      buffer = chunk
      continue
    }

    const graphemeCount = [...buffer].length
    if (graphemeCount < MIN_CHUNK_GRAPHEMES) {
      buffer = buffer + ' ' + chunk
    } else {
      merged.push(buffer)
      buffer = chunk
    }
  }

  if (buffer.length > 0) {
    merged.push(buffer)
  }

  return merged
}

function splitByParagraphs (text) {
  return text.split(/\n\s*\n/).map(p => p.trim()).filter(p => p.length > 0)
}

function splitText (text) {
  const paragraphs = splitByParagraphs(text)
  const allChunks = []

  for (const paragraph of paragraphs) {
    const sentences = splitBySentence(paragraph)
    const bounded = []
    for (const s of sentences) {
      const parts = splitLongChunk(s)
      for (const p of parts) {
        bounded.push(p)
      }
    }
    const merged = mergeShortChunks(bounded)
    for (const chunk of merged) {
      if (chunk.length > 0) {
        allChunks.push(chunk)
      }
    }
  }

  if (allChunks.length === 0 && text.trim().length > 0) {
    return splitLongChunk(text.trim())
  }

  return allChunks
}

module.exports = {
  splitText,
  splitBySentence,
  splitByParagraphs,
  mergeShortChunks,
  splitLongChunk,
  splitOnWordBoundary,
  MAX_CHUNK_GRAPHEMES
}
