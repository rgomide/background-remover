/** Models that use ONNX U-2-Net style I/O (manual preprocess, no U2NetImageProcessor in Transformers.js yet). */
export const U2NET_SUPPORTED_MODEL_IDS = new Set(['BritishWerewolf/U-2-Net']);

export function isU2NetModelId(modelId) {
  return U2NET_SUPPORTED_MODEL_IDS.has(String(modelId || '').trim());
}

/** Filename token for `getUniqueOutputPath` (`*_rmbg_no_bg.png`, etc.). */
export function backgroundRemovalOutputFileTag(modelId) {
  const id = String(modelId || '');
  if (isU2NetModelId(id)) return 'u2net_no_bg';
  if (id.toLowerCase().includes('modnet')) return 'modnet_no_bg';
  return 'rmbg_no_bg';
}

/** Hub repo only ships `onnx/model.onnx` (fp32); there is no `model_quantized.onnx` for q8. */
const warnedU2NetDtypes = new Set();

export function normalizeU2NetDtype(requested) {
  const r = String(requested ?? 'fp32').trim().toLowerCase();
  if (r === 'fp32' || r === 'float32') return 'fp32';
  if (!warnedU2NetDtypes.has(r)) {
    warnedU2NetDtypes.add(r);
    console.warn(
      `[BritishWerewolf/U-2-Net] Only fp32 weights exist (\`onnx/model.onnx\`). Ignoring RMBG_DTYPE="${requested}"; using fp32.`,
    );
  }
  return 'fp32';
}

const SIZE = 320;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

function u2netPreprocessMode() {
  const m = String(process.env.U2NET_PREPROCESS ?? 'rembg').trim().toLowerCase();
  if (m === 'hf' || m === 'letterbox' || m === 'hub') return 'hf';
  return 'rembg';
}

/**
 * HuggingFace-style: longest edge 320, center-pad to 320², /255, ImageNet norm (matches Hub `preprocessor_config.json`).
 * Only use if `U2NET_PREPROCESS=hf` — geometry must match how the ONNX was exported.
 */
async function preprocessU2NetPixelValuesLetterbox(rawImage, Tensor) {
  let img = rawImage.clone().rgb();
  const [iw, ih] = img.size;
  const scale = SIZE / Math.max(iw, ih);
  const nw = Math.max(1, Math.round(iw * scale));
  const nh = Math.max(1, Math.round(ih * scale));
  img = await img.resize(nw, nh);

  const padLeft = Math.floor((SIZE - nw) / 2);
  const padTop = Math.floor((SIZE - nh) / 2);

  const data = new Float32Array(3 * SIZE * SIZE);
  for (let py = 0; py < SIZE; py++) {
    for (let px = 0; px < SIZE; px++) {
      let r = 0;
      let g = 0;
      let b = 0;
      const sx = px - padLeft;
      const sy = py - padTop;
      if (sx >= 0 && sx < nw && sy >= 0 && sy < nh) {
        const idx = (sy * nw + sx) * 3;
        r = img.data[idx];
        g = img.data[idx + 1];
        b = img.data[idx + 2];
      }
      const channels = [r, g, b];
      for (let c = 0; c < 3; c++) {
        const norm = (channels[c] / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        data[c * SIZE * SIZE + py * SIZE + px] = norm;
      }
    }
  }

  return new Tensor('float32', data, [1, 3, SIZE, SIZE]);
}

/**
 * rembg `BaseSession.normalize` (used by `U2netSession`): RGB → **exact 320×320 stretch** (LANCZOS),
 * divide all channels by **global max**, then ImageNet mean/std, NCHW.
 * Matches the usual `u2net.onnx` path from [rembg](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/base.py)
 * and avoids mask/content misalignment from letterboxing.
 */
async function preprocessU2NetPixelValuesRembg(rawImage, Tensor) {
  let img = rawImage.clone().rgb();
  img = await img.resize(SIZE, SIZE, { resample: 'lanczos' });

  let mx = 1e-6;
  const flat = img.data;
  for (let i = 0; i < flat.length; i++) {
    if (flat[i] > mx) mx = flat[i];
  }
  const invMx = 1 / mx;

  const data = new Float32Array(3 * SIZE * SIZE);
  for (let py = 0; py < SIZE; py++) {
    for (let px = 0; px < SIZE; px++) {
      const idx = (py * SIZE + px) * 3;
      const r = flat[idx] * invMx;
      const g = flat[idx + 1] * invMx;
      const b = flat[idx + 2] * invMx;
      data[0 * SIZE * SIZE + py * SIZE + px] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      data[1 * SIZE * SIZE + py * SIZE + px] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      data[2 * SIZE * SIZE + py * SIZE + px] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
  }

  return new Tensor('float32', data, [1, 3, SIZE, SIZE]);
}

export async function preprocessU2NetPixelValues(rawImage) {
  const { Tensor } = await import('@huggingface/transformers');
  const mode = u2netPreprocessMode();
  if (mode === 'hf') {
    return preprocessU2NetPixelValuesLetterbox(rawImage, Tensor);
  }
  return preprocessU2NetPixelValuesRembg(rawImage, Tensor);
}


/**
 * If values look like logits, apply sigmoid so we are in ~[0,1].
 * @param {Float32Array|import('@huggingface/transformers').TypedArray} floatData
 */
function logitsToProb01(floatData) {
  const epsilon = 1e-5;
  const arr = floatData instanceof Float32Array ? new Float32Array(floatData) : new Float32Array(floatData);
  let needsSigmoid = false;
  for (let i = 0; i < arr.length; i++) {
    const x = arr[i];
    if (x < -epsilon || x > 1 + epsilon) {
      needsSigmoid = true;
      break;
    }
  }
  if (needsSigmoid) {
    for (let i = 0; i < arr.length; i++) {
      const x = arr[i];
      arr[i] = 1 / (1 + Math.exp(-x));
    }
  }
  return arr;
}

/**
 * Stretch dynamic range to [0,1] (same idea as rembg's U2netSession: min–max then ×255).
 * Without this, ONNX saliency often sits in a narrow band so alpha looks "foggy" / blurred.
 */
function minMaxStretch01(floatData) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < floatData.length; i++) {
    const v = floatData[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min;
  if (!(range > 1e-8)) {
    return new Float32Array(floatData.length).fill(0);
  }
  const out = new Float32Array(floatData.length);
  const inv = 1 / range;
  for (let i = 0; i < floatData.length; i++) {
    out[i] = (floatData[i] - min) * inv;
  }
  return out;
}

const u2netModelPromises = new Map();

/** Transformers.js logs a noisy `console.warn` for `model_type: u2net` even though `AutoModel` works. */
let u2netWarnFilterDepth = 0;
let savedConsoleWarn = null;
let loggedU2NetAutoModelNote = false;

function isU2NetUnknownClassWarn(args) {
  if (!args.length || typeof args[0] !== 'string') return false;
  const msg = args[0];
  return (
    msg.includes('Unknown model class') &&
    msg.includes('u2net') &&
    msg.includes('attempting to construct from base class')
  );
}

async function loadU2NetAutoModel(AutoModel, modelId, resolved) {
  u2netWarnFilterDepth += 1;
  if (u2netWarnFilterDepth === 1) {
    savedConsoleWarn = console.warn;
    console.warn = (...args) => {
      if (isU2NetUnknownClassWarn(args)) return;
      savedConsoleWarn.apply(console, args);
    };
  }
  try {
    if (!loggedU2NetAutoModelNote) {
      loggedU2NetAutoModelNote = true;
      console.log(
        '[BritishWerewolf/U-2-Net] Loading ONNX via AutoModel (Transformers.js has no dedicated U2Net class; this is expected).',
      );
    }
    return await AutoModel.from_pretrained(modelId, resolved);
  } finally {
    u2netWarnFilterDepth -= 1;
    if (u2netWarnFilterDepth === 0 && savedConsoleWarn) {
      console.warn = savedConsoleWarn;
      savedConsoleWarn = null;
    }
  }
}

function u2netModelCacheKey(modelId, loadOptions) {
  const dtype = normalizeU2NetDtype(loadOptions?.dtype);
  const so = loadOptions?.session_options;
  const soKey = so && typeof so === 'object' ? JSON.stringify(so) : '';
  return `${modelId}::${dtype}::${soKey}`;
}

export function getU2NetModelPromise(modelId, loadOptions) {
  const resolved = {
    ...loadOptions,
    dtype: normalizeU2NetDtype(loadOptions?.dtype),
  };
  const key = u2netModelCacheKey(modelId, resolved);
  let p = u2netModelPromises.get(key);
  if (!p) {
    p = (async () => {
      const { AutoModel } = await import('@huggingface/transformers');
      return loadU2NetAutoModel(AutoModel, modelId, resolved);
    })();
    u2netModelPromises.set(key, p);
  }
  return p;
}

export function resetU2NetModelCache() {
  u2netModelPromises.clear();
  loggedU2NetAutoModelNote = false;
}

/**
 * @param {import('@huggingface/transformers').RawImage} inputImage  RGB or RGBA; not mutated
 * @param {string} modelId
 * @param {Record<string, unknown>} loadOptions e.g. `{ dtype: 'fp32', session_options: {} }`
 */
export async function runU2NetBackgroundRemoval(inputImage, modelId, loadOptions) {
  const { RawImage } = await import('@huggingface/transformers');
  const original = inputImage.clone().rgba();
  const [ow, oh] = original.size;

  const pixelValues = await preprocessU2NetPixelValues(inputImage);
  const resolvedOptions = {
    ...loadOptions,
    dtype: normalizeU2NetDtype(loadOptions?.dtype),
  };
  const model = await getU2NetModelPromise(modelId, resolvedOptions);
  const inputKey = model.config?.input_name?.[0] ?? 'input.1';
  const feed = { [inputKey]: pixelValues };
  const output = await model(feed);

  const compositeKey = String(model.config?.output_composite ?? '1959');
  const maskTensor = output[compositeKey];
  if (!maskTensor?.data) {
    throw new Error(`U-2-Net output "${compositeKey}" missing from model forward pass`);
  }

  const skipStretch =
    process.env.U2NET_SKIP_MASK_MINMAX === '1' || process.env.U2NET_SKIP_MASK_MINMAX === 'true';
  let probs = logitsToProb01(maskTensor.data);
  probs = skipStretch ? probs : minMaxStretch01(probs);

  const mh = maskTensor.dims[maskTensor.dims.length - 2];
  const mw = maskTensor.dims[maskTensor.dims.length - 1];
  const u8 = new Uint8ClampedArray(mw * mh);
  for (let i = 0; i < u8.length; i++) {
    u8[i] = Math.round(Math.min(1, Math.max(0, probs[i])) * 255);
  }

  let maskImage = new RawImage(u8, mw, mh, 1);
  if (ow !== mw || oh !== mh) {
    maskImage = await maskImage.resize(ow, oh, { resample: 'lanczos' });
  }

  original.putAlpha(maskImage);
  return original;
}
