import { pipeline, RawImage, env } from '@huggingface/transformers';
import fs from 'fs';
import path from 'path';

// Disable local model lookup if the model is not downloaded yet.
env.allowLocalModels = false;

// If you see "Unauthorized access to file", create a Read token at:
// https://huggingface.co/settings/tokens
// Then run: HF_TOKEN=hf_xxxxxxxx node image.js <URL>

const RMBG_MODEL = 'briaai/RMBG-2.0';
const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif']);
const RMBG_DTYPE = process.env.RMBG_DTYPE ?? 'fp32';
const RMBG_MAX_SIDE = Number.parseInt(process.env.RMBG_MAX_SIDE ?? '0', 10);

function suppressOnnxShapeReuseWarnings() {
  const originalWrite = process.stderr.write.bind(process.stderr);
  process.stderr.write = (chunk, encoding, callback) => {
    const text = typeof chunk === 'string' ? chunk : chunk?.toString?.() ?? '';
    if (text.includes('AllocateMLValueTensorPreAllocateBuffer') && text.includes('Shape mismatch attempting to re-use buffer')) {
      if (typeof callback === 'function') callback();
      return true;
    }
    return originalWrite(chunk, encoding, callback);
  };
}

/** For briaai/RMBG-2.0, HF config.json may ship with model_type: null. */
/** The library requires model_type, so we patch it before loading. */
async function getConfigForModel(modelId) {
  const token = process.env.HF_TOKEN ?? process.env.HF_ACCESS_TOKEN;
  const url = `https://huggingface.co/${modelId}/resolve/main/config.json`;
  const headers = { 'User-Agent': 'transformers.js-custom' };
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(url, { headers });
  if (!res.ok) return null;
  const config = await res.json();
  if ((modelId === 'briaai/RMBG-2.0' || modelId === 'briaai/RMBG-1.4') && config.model_type == null) {
    config.model_type = 'birefnet';
  }
  return config;
}

function isHttpUrl(value) {
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

async function collectImageFilesFromDirectory(directoryPath) {
  const entries = await fs.promises.readdir(directoryPath, { withFileTypes: true });
  const collected = [];

  for (const entry of entries) {
    const fullPath = path.join(directoryPath, entry.name);
    if (entry.isDirectory()) {
      collected.push(...await collectImageFilesFromDirectory(fullPath));
      continue;
    }

    if (entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      collected.push(fullPath);
    }
  }

  return collected;
}

async function resolveSources(input) {
  if (isHttpUrl(input)) {
    return [input];
  }

  const resolvedPath = path.resolve(input);
  let stats;
  try {
    stats = await fs.promises.stat(resolvedPath);
  } catch {
    throw new Error(`Source does not exist: ${input}`);
  }

  if (stats.isFile()) {
    if (!IMAGE_EXTENSIONS.has(path.extname(resolvedPath).toLowerCase())) {
      throw new Error(`Unsupported file extension: ${path.extname(resolvedPath) || '(none)'}`);
    }
    return [resolvedPath];
  }

  if (stats.isDirectory()) {
    const files = await collectImageFilesFromDirectory(resolvedPath);
    if (files.length === 0) {
      throw new Error(`No supported image files found in folder: ${resolvedPath}`);
    }
    return files;
  }

  throw new Error(`Unsupported source type: ${input}`);
}

function getUniqueOutputPath(source, executionTimeMs) {
  const sourceName = isHttpUrl(source)
    ? 'resultado_url'
    : path.parse(source).name.replace(/[^a-zA-Z0-9-_]/g, '_');
  const baseName = `${sourceName}-execution_time_${executionTimeMs}ms`;

  let counter = 1;
  let candidate = `${baseName}_rmbg_no_bg.png`;
  while (fs.existsSync(candidate)) {
    candidate = `${baseName}_rmbg_no_bg_${counter}.png`;
    counter += 1;
  }
  return candidate;
}

async function maybeResizeForSpeed(image) {
  if (!Number.isInteger(RMBG_MAX_SIDE) || RMBG_MAX_SIDE <= 0) {
    return image;
  }

  const [width, height] = image.size;
  const currentMaxSide = Math.max(width, height);
  if (currentMaxSide <= RMBG_MAX_SIDE) {
    return image;
  }

  const scale = RMBG_MAX_SIDE / currentMaxSide;
  const resizedWidth = Math.max(1, Math.round(width * scale));
  const resizedHeight = Math.max(1, Math.round(height * scale));

  console.log(`Resizing ${width}x${height} -> ${resizedWidth}x${resizedHeight} for faster inference`);
  return image.resize(resizedWidth, resizedHeight);
}

async function processImage() {
  suppressOnnxShapeReuseWarnings();

  const source = process.argv[2];

  if (!source) {
    console.error("Usage: node image.js <source>");
    console.error("source can be: image file path, folder path, or image URL");
    process.exit(1);
  }

  try {
    const processStart = performance.now();
    console.log(`--- Initializing model (${RMBG_MODEL}, dtype=${RMBG_DTYPE}) ---`);
    const config = await getConfigForModel(RMBG_MODEL);
    const pipelineOptions = { dtype: RMBG_DTYPE };
    if (config) pipelineOptions.config = config;
    const segmenter = await pipeline('background-removal', RMBG_MODEL, pipelineOptions);

    const sources = await resolveSources(source);
    console.log(`--- Processing ${sources.length} image(s) ---`);

    for (const [index, imageSource] of sources.entries()) {
      const imageStart = performance.now();
      console.log(`[${index + 1}/${sources.length}] ${imageSource}`);
      const image = await RawImage.fromURL(imageSource);
      const inputImage = await maybeResizeForSpeed(image);

      // background-removal returns a RawImage array (transparent background result)
      const results = await segmenter(inputImage);
      const resultImage = Array.isArray(results) ? results[0] : results;

      const imageElapsedMs = Math.round(performance.now() - imageStart);
      const outputPath = getUniqueOutputPath(imageSource, imageElapsedMs);
      await resultImage.save(outputPath);
      console.log(`Saved: ${outputPath} (${imageElapsedMs} ms)`);
    }

    const totalElapsedMs = Math.round(performance.now() - processStart);
    console.log(`--- Done in ${totalElapsedMs} ms ---`);

  } catch (error) {
    console.error("\nERROR:");
    console.error(error.message);

    if (error.message.includes('Unauthorized')) {
      console.log("\nTIP: Hugging Face may require authentication to download the model.");
      console.log("1. Create a Read token at https://huggingface.co/settings/tokens.");
      console.log("2. Run: HF_TOKEN=your_token node image.js <source>");
    }
    if (error.message.includes('Forbidden')) {
      console.log("\nTIP: RMBG-2.0 is gated, so you must accept the license on Hugging Face.");
      console.log("1. Visit https://huggingface.co/briaai/RMBG-2.0");
      console.log("2. Sign in with the same account used by HF_TOKEN.");
      console.log("3. Accept the model license on the page.");
      console.log("4. Run the command again.");
      console.log("\nAlternative: use 1.4 (not gated) by changing the script to 'briaai/RMBG-1.4'.");
    }
    if (error.message.includes('Unsupported model type')) {
      console.log("\nTIP: Check whether HF_TOKEN is set in .env and if you accepted the license at https://huggingface.co/briaai/RMBG-2.0");
      console.log("Alternative: change RMBG_MODEL in the script to 'briaai/RMBG-1.4'.");
    }
  }
}

processImage();