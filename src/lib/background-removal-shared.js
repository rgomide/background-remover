import { getConfigForModel } from './huggingface-model-config.js';
import { isU2NetModelId, runU2NetBackgroundRemoval } from './u2net-infer.js';

let cachedPipelineKey = null;
let cachedPipelinePromise = null;

function pipelineCacheKey({ modelId, dtype, session_options, userAgent }) {
  return JSON.stringify({ modelId, dtype, session_options, userAgent });
}

/**
 * @param {import('@huggingface/transformers').RawImage} inputImage
 * @param {{ modelId: string, dtype: string, session_options: Record<string, unknown>, userAgent: string }} options
 */
export async function inferBackgroundRemovalRawImage(inputImage, options) {
  const { modelId, dtype, session_options, userAgent } = options;

  if (isU2NetModelId(modelId)) {
    const { env } = await import('@huggingface/transformers');
    env.allowLocalModels = false;
    return runU2NetBackgroundRemoval(inputImage, modelId, { dtype, session_options });
  }

  const key = pipelineCacheKey(options);
  if (!cachedPipelinePromise || cachedPipelineKey !== key) {
    cachedPipelineKey = key;
    cachedPipelinePromise = (async () => {
      const { pipeline, env } = await import('@huggingface/transformers');
      env.allowLocalModels = false;
      const config = await getConfigForModel(modelId, { userAgent });
      const pipelineOptions = { dtype, session_options };
      if (config) pipelineOptions.config = config;
      return pipeline('background-removal', modelId, pipelineOptions);
    })();
  }

  const segmenter = await cachedPipelinePromise;
  const results = await segmenter(inputImage);
  return Array.isArray(results) ? results[0] : results;
}
