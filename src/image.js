import { RawImage, env } from '@huggingface/transformers';
import fs from 'node:fs';
import path from 'node:path';
import { inferBackgroundRemovalRawImage } from './lib/background-removal-shared.js';
import { parseEnvIntNonNegative, parseEnvStringOr, parseProcessPriorityWithDefault } from './lib/env.js';
import { maybeResizeForSpeed } from './lib/image-resize.js';
import { suppressOnnxShapeReuseWarnings } from './lib/onnx-warnings.js';
import { trySetProcessPriority } from './lib/process-priority.js';
import { backgroundRemovalOutputFileTag } from './lib/u2net-infer.js';
import { getUniqueOutputPath, resolveSources } from './lib/cli-image-sources.js';

env.allowLocalModels = false;

const RMBG_MODEL = parseEnvStringOr('RMBG_MODEL', 'briaai/RMBG-2.0');
const RMBG_DTYPE = parseEnvStringOr('RMBG_DTYPE', 'fp32');
const RMBG_MAX_SIDE = parseEnvIntNonNegative('RMBG_MAX_SIDE', 0);
const OUTPUT_DIR = path.resolve(process.cwd(), 'out');
const PROCESS_PRIORITY = parseProcessPriorityWithDefault(-10);

async function processImage() {
  suppressOnnxShapeReuseWarnings();
  trySetProcessPriority(PROCESS_PRIORITY);

  const source = process.argv[2];

  if (!source) {
    console.error('Usage: npm run image -- <source>');
    console.error('source can be: image file path, folder path, or image URL');
    process.exit(1);
  }

  try {
    const processStart = performance.now();
    await fs.promises.mkdir(OUTPUT_DIR, { recursive: true });
    console.log(`--- Model ${RMBG_MODEL} (dtype=${RMBG_DTYPE}) — first image may download weights ---`);

    const sources = await resolveSources(source);
    console.log(`--- Processing ${sources.length} image(s) ---`);

    for (const [index, imageSource] of sources.entries()) {
      const imageStart = performance.now();
      console.log(`[${index + 1}/${sources.length}] ${imageSource}`);
      const image = await RawImage.fromURL(imageSource);
      const inputImage = await maybeResizeForSpeed(image, RMBG_MAX_SIDE);

      const resultImage = await inferBackgroundRemovalRawImage(inputImage, {
        modelId: RMBG_MODEL,
        dtype: RMBG_DTYPE,
        session_options: {},
        userAgent: 'transformers.js-custom',
      });

      const imageElapsedMs = Math.round(performance.now() - imageStart);
      const outputPath = getUniqueOutputPath(imageSource, imageElapsedMs, OUTPUT_DIR, {
        fileTag: backgroundRemovalOutputFileTag(RMBG_MODEL),
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
      console.log('\nTIP: Hugging Face may require authentication to download the model.');
      console.log('1. Create a Read token at https://huggingface.co/settings/tokens.');
      console.log('2. Run: HF_TOKEN=your_token npm run image -- <source>');
    }
    if (error.message.includes('Forbidden')) {
      console.log('\nTIP: RMBG-2.0 is gated, so you must accept the license on Hugging Face.');
      console.log('1. Visit https://huggingface.co/briaai/RMBG-2.0');
      console.log('2. Sign in with the same account used by HF_TOKEN.');
      console.log('3. Accept the model license on the page.');
      console.log('4. Run the command again.');
      console.log("\nAlternative: use 1.4 (not gated) by changing RMBG_MODEL in the script to 'briaai/RMBG-1.4'.");
    }
    if (error.message.includes('Unsupported model type')) {
      console.log(
        '\nTIP: Check whether HF_TOKEN is set in .env and if you accepted the license at https://huggingface.co/briaai/RMBG-2.0',
      );
      console.log("Alternative: change RMBG_MODEL in the script to 'briaai/RMBG-1.4'.");
    }
  }
}

processImage();
