import { pipeline, RawImage, env } from '@huggingface/transformers';
import fs from 'fs';

// Disable local model lookup if the model is not downloaded yet.
env.allowLocalModels = false;

// Apache 2.0 model (commercial use allowed). Xenova/modnet is portrait-focused
// and offers solid performance for productivity use cases.
const BACKGROUND_REMOVAL_MODEL = 'Xenova/modnet';

async function processImage() {
  const imageUrl = process.argv[2];

  if (!imageUrl) {
    console.error("Usage: node imageFreeLicence.js <URL>");
    process.exit(1);
  }

  try {
    console.log("--- Initializing model (Xenova/modnet, Apache 2.0) ---");
    const segmenter = await pipeline('background-removal', BACKGROUND_REMOVAL_MODEL, { dtype: 'fp32' });

    console.log("--- Processing image ---");
    const image = await RawImage.fromURL(imageUrl);

    // background-removal returns a RawImage array (transparent background result)
    const results = await segmenter(image);
    const resultImage = Array.isArray(results) ? results[0] : results;

    let counter = 1;
    while (fs.existsSync(`resultado_${counter}.png`)) { counter++; }
    const fileName = `resultado_${counter}.png`;

    await resultImage.save(fileName);

    console.log(`--- Done! Saved as: ${fileName} ---`);

  } catch (error) {
    console.error("\nERROR:");
    console.error(error.message);

    if (error.message.includes('Unauthorized')) {
      console.log("\nTIP: Create a Read token at https://huggingface.co/settings/tokens and run:");
      console.log("  HF_TOKEN=your_token node imageFreeLicence.js <URL>");
    }
  }
}

processImage();