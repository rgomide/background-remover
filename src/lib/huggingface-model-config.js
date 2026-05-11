/**
 * For briaai/RMBG-2.0 / RMBG-1.4, HF config.json may ship with model_type: null.
 * The library requires model_type, so we patch it before loading.
 */
export async function getConfigForModel(modelId, options = {}) {
  const userAgent = options.userAgent ?? 'background-remover';
  const token = process.env.HF_TOKEN ?? process.env.HF_ACCESS_TOKEN;
  const url = `https://huggingface.co/${modelId}/resolve/main/config.json`;
  const headers = { 'User-Agent': userAgent };
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(url, { headers });
  if (!res.ok) return null;
  const config = await res.json();
  if ((modelId === 'briaai/RMBG-2.0' || modelId === 'briaai/RMBG-1.4') && config.model_type == null) {
    config.model_type = 'birefnet';
  }
  return config;
}
