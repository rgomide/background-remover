/** Empty `VAR=` in `.env` yields `""`, which `??` does not replace — avoid NaN from `parseInt("")`. */

export function parseEnvPort(name, fallback) {
  const raw = process.env[name];
  if (raw == null || String(raw).trim() === '') return fallback;
  const n = Number.parseInt(String(raw), 10);
  if (!Number.isInteger(n) || n <= 0 || n > 65535) return fallback;
  return n;
}

export function parseEnvIntNonNegative(name, fallback) {
  const raw = process.env[name];
  if (raw == null || String(raw).trim() === '') return fallback;
  const n = Number.parseInt(String(raw), 10);
  if (!Number.isInteger(n) || n < 0) return fallback;
  return n;
}

/** Server: only set niceness when explicitly configured. */
export function parseProcessPriorityOptional() {
  const raw = process.env.PROCESS_PRIORITY;
  if (raw == null || String(raw).trim() === '') return null;
  const n = Number.parseInt(String(raw), 10);
  return Number.isInteger(n) ? n : null;
}

/** CLI scripts: keep a numeric default when the variable is unset (legacy behavior). */
export function parseProcessPriorityWithDefault(defaultValue) {
  const raw = process.env.PROCESS_PRIORITY;
  if (raw == null || String(raw).trim() === '') return defaultValue;
  const n = Number.parseInt(String(raw), 10);
  return Number.isInteger(n) ? n : defaultValue;
}

export function parseEnvStringOr(name, fallback) {
  const raw = process.env[name];
  if (raw == null) return fallback;
  const t = String(raw).trim();
  return t === '' ? fallback : t;
}
