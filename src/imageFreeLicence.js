import { pipeline, RawImage, env } from '@huggingface/transformers';
import fs from 'node:fs';
import path from 'node:path';
import { parseEnvIntNonNegative, parseProcessPriorityWithDefault } from './lib/env.js';
import { maybeResizeForSpeed } from './lib/image-resize.js';
import { suppressOnnxShapeReuseWarnings } from './lib/onnx-warnings.js';
import { trySetProcessPriority } from './lib/process-priority.js';
import { getUniqueOutputPath, resolveSources } from './lib/cli-image-sources.js';

env.allowLocalModels = false;

const BACKGROUND_REMOVAL_MODEL = 'Xenova/modnet';
const RMBG_MAX_SIDE = parseEnvIntNonNegative('RMBG_MAX_SIDE', 0);
const OUTPUT_DIR = path.resolve(process.cwd(), 'out');
const PROCESS_PRIORITY = parseProcessPriorityWithDefault(-10);

async function processImage() {
  suppressOnnxShapeReuseWarnings();
  trySetProcessPriority(PROCESS_PRIORITY);

  const source = process.argv[2];

  if (!source) {
    console.error('Usage: npm run imageFreeLicence -- <source>');
    console.error('source can be: image file path, folder path, or image URL');
    process.exit(1);
  }

  try {
    const processStart = performance.now();
    await fs.promises.mkdir(OUTPUT_DIR, { recursive: true });
    console.log('--- Initializing model (Xenova/modnet, Apache 2.0) ---');
    const segmenter = await pipeline('background-removal', BACKGROUND_REMOVAL_MODEL, { dtype: 'fp32' });

    const sources = await resolveSources(source);
    console.log(`--- Processing ${sources.length} image(s) ---`);

    for (const [index, imageSource] of sources.entries()) {
      const imageStart = performance.now();
      console.log(`[${index + 1}/${sources.length}] ${imageSource}`);
      const image = await RawImage.fromURL(imageSource);
      const inputImage = await maybeResizeForSpeed(image, RMBG_MAX_SIDE);

      const results = await segmenter(inputImage);
      const resultImage = Array.isArray(results) ? results[0] : results;

      const imageElapsedMs = Math.round(performance.now() - imageStart);
      const outputPath = getUniqueOutputPath(imageSource, imageElapsedMs, OUTPUT_DIR, {
        fileTag: 'modnet_no_bg',
      });
      await resultImage.save(outputPath);
      console.log(`Saved: ${outputPath} (${imageElapsedMs} ms)`);
    }

    const totalElapsedMs = Math.round(performance.now() - processStart);
    console.log(`--- Done in ${totalElapsedMs} ms ---`);
  } catch (error) {
    console.error('\nERROR:');
    console.error(error.message);

    if (error.message.includes('Unauthorized')) {
      console.log('\nTIP: Create a Read token at https://huggingface.co/settings/tokens and run:');
      console.log('  HF_TOKEN=your_token npm run imageFreeLicence -- <source>');
    }
  }
}

processImage();
