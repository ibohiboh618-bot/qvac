export async function loadModel() {
  return 'mock-model-id';
}

export async function* completion() {
  yield '[]';
}

export const LLAMA_3_2_1B_INST_Q4_0 = 'llama-3.2-1b-instruct-q4_0';
