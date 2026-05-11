import { parseEnvIntNonNegative } from './env.js';

/**
 * @returns {'low' | 'high'}
 * - `low`: smaller ONNX session footprint (Render free tier, ~512MB RAM).
 * - `high`: library defaults (local dev / larger instances).
 *
 * Explicit `TRANSFORMERS_MEMORY_PROFILE=low|high` wins. When unset, `RENDER=true`
 * or `LOW_MEMORY=1` selects `low`.
 */
export function getTransformersMemoryProfile() {
  const raw = process.env.TRANSFORMERS_MEMORY_PROFILE;
  if (raw != null && String(raw).trim() !== '') {
    const v = String(raw).trim().toLowerCase();
    if (v === 'low' || v === 'minimal') return 'low';
    return 'high';
  }
  if (process.env.RENDER === 'true') return 'low';
  if (process.env.LOW_MEMORY === '1' || process.env.LOW_MEMORY === 'true') return 'low';
  return 'high';
}

/**
 * Session options forwarded to `pipeline(..., { session_options })` for onnxruntime-node.
 * @see https://onnxruntime.ai/docs/api/js/interfaces/InferenceSession.SessionOptions.html
 */
export function getOnnxSessionOptionsForPipeline() {
  if (getTransformersMemoryProfile() !== 'low') {
    return {};
  }
  const intraOpNumThreads = Math.max(1, parseEnvIntNonNegative('ORT_INTRA_OP_THREADS', 1));
  const interOpNumThreads = Math.max(1, parseEnvIntNonNegative('ORT_INTER_OP_THREADS', 1));
  return {
    intraOpNumThreads,
    interOpNumThreads,
    executionMode: 'sequential',
    enableCpuMemArena: false,
    enableMemPattern: false,
  };
}
