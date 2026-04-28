import { pipeline, RawImage, env } from '@huggingface/transformers';
import fs from 'fs';

// Disable local model lookup if the model is not downloaded yet.
env.allowLocalModels = false;

// If you see "Unauthorized access to file", create a Read token at:
// https://huggingface.co/settings/tokens
// Then run: HF_TOKEN=hf_xxxxxxxx node image.js <URL>

const RMBG_MODEL = 'briaai/RMBG-2.0';

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

async function processImage() {
  const imageUrl = process.argv[2];

  if (!imageUrl) {
    console.error("Usage: node image.js <URL>");
    process.exit(1);
  }

  try {
    console.log("--- Initializing model ---");
    const config = await getConfigForModel(RMBG_MODEL);
    const pipelineOptions = config ? { config } : {};
    const segmenter = await pipeline('background-removal', RMBG_MODEL, pipelineOptions);

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
      console.log("\nTIP: Hugging Face may require authentication to download the model.");
      console.log("1. Create a Read token at https://huggingface.co/settings/tokens.");
      console.log("2. Run: HF_TOKEN=your_token node image.js <URL>");
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