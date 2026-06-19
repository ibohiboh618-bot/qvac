/*
 * Minimal GitHub Actions toolkit shim.
 *
 * Replaces @actions/core for the handful of helpers this action uses. Avoiding
 * the dependency keeps the bundle to first-party code only: @actions/core@3 is
 * ESM-only and drags in @actions/http-client -> undici, whose vendored
 * WebSocket SHA-1 handshake trips CodeQL's "weak crypto" query when bundled.
 *
 * Implements the same workflow-command protocol as the toolkit
 * (https://github.com/actions/toolkit/blob/main/packages/core).
 */

interface InputOptions {
  required?: boolean
}

function escapeData(value: string): string {
  return value.replace(/%/g, '%25').replace(/\r/g, '%0D').replace(/\n/g, '%0A')
}

function issueCommand(command: string, message: string): void {
  process.stdout.write(`::${command}::${escapeData(message)}\n`)
}

export function getInput(name: string, options?: InputOptions): string {
  const value = process.env[`INPUT_${name.replace(/ /g, '_').toUpperCase()}`] || ''
  if (options && options.required && !value) {
    throw new Error(`Input required and not supplied: ${name}`)
  }
  return value.trim()
}

export function info(message: string): void {
  process.stdout.write(`${message}\n`)
}

export function warning(message: string): void {
  issueCommand('warning', message)
}

export function error(message: string): void {
  issueCommand('error', message)
}

export function setFailed(message: string): void {
  process.exitCode = 1
  error(message)
}
