import { pipeline, RawImage, env } from '@huggingface/transformers';
import fs from 'fs';
import path from 'path';

// Disable local model lookup if the model is not downloaded yet.
env.allowLocalModels = false;

// Apache 2.0 model (commercial use allowed). Xenova/modnet is portrait-focused
// and offers solid performance for productivity use cases.
const BACKGROUND_REMOVAL_MODEL = 'Xenova/modnet';
const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tiff', '.tif']);

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
  let candidate = `${baseName}_modnet_no_bg.png`;
  while (fs.existsSync(candidate)) {
    candidate = `${baseName}_modnet_no_bg_${counter}.png`;
    counter += 1;
  }
  return candidate;
}

async function processImage() {
  suppressOnnxShapeReuseWarnings();

  const source = process.argv[2];

  if (!source) {
    console.error("Usage: node imageFreeLicence.js <source>");
    console.error("source can be: image file path, folder path, or image URL");
    process.exit(1);
  }

  try {
    const processStart = performance.now();
    console.log("--- Initializing model (Xenova/modnet, Apache 2.0) ---");
    const segmenter = await pipeline('background-removal', BACKGROUND_REMOVAL_MODEL, { dtype: 'fp32' });

    const sources = await resolveSources(source);
    console.log(`--- Processing ${sources.length} image(s) ---`);

    for (const [index, imageSource] of sources.entries()) {
      const imageStart = performance.now();
      console.log(`[${index + 1}/${sources.length}] ${imageSource}`);
      const image = await RawImage.fromURL(imageSource);

      // background-removal returns a RawImage array (transparent background result)
      const results = await segmenter(image);
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
      console.log("\nTIP: Create a Read token at https://huggingface.co/settings/tokens and run:");
      console.log("  HF_TOKEN=your_token node imageFreeLicence.js <source>");
    }
  }
}

processImage();