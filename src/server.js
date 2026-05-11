import express from 'express';
import { createServer } from 'node:http';
import {
  parseEnvIntNonNegative,
  parseEnvPort,
  parseEnvStringOr,
  parseProcessPriorityOptional,
} from './lib/env.js';
import { getConfigForModel } from './lib/huggingface-model-config.js';
import { maybeResizeForSpeed } from './lib/image-resize.js';
import { suppressOnnxShapeReuseWarnings } from './lib/onnx-warnings.js';
import { trySetProcessPriority } from './lib/process-priority.js';
import { assertRemoteImageUrlAllowed } from './lib/ssrf-url.js';
import {
  getOnnxSessionOptionsForPipeline,
  getTransformersMemoryProfile,
} from './lib/transformers-memory-profile.js';

const PORT = parseEnvPort('PORT', 3000);
const HOST = (() => {
  const raw = process.env.HOST;
  if (raw == null || String(raw).trim() === '') return '0.0.0.0';
  return String(raw).trim();
})();
const RMBG_MODEL = process.env.RMBG_MODEL ?? 'Xenova/modnet';
const RMBG_DTYPE = parseEnvStringOr('RMBG_DTYPE', 'fp32');
const RMBG_MAX_SIDE = parseEnvIntNonNegative('RMBG_MAX_SIDE', 0);
const IMAGE_FETCH_TIMEOUT_MS = parseEnvIntNonNegative('IMAGE_FETCH_TIMEOUT_MS', 30000);
const PROCESS_PRIORITY = parseProcessPriorityOptional();

let transformersModulePromise = null;

async function getTransformers() {
  if (!transformersModulePromise) {
    transformersModulePromise = import('@huggingface/transformers').then((mod) => {
      mod.env.allowLocalModels = false;
      return mod;
    });
  }
  return transformersModulePromise;
}

suppressOnnxShapeReuseWarnings();
trySetProcessPriority(PROCESS_PRIORITY);

async function loadImageFromRemoteUrl(imageUrlString) {
  await assertRemoteImageUrlAllowed(imageUrlString);
  const signal =
    Number.isFinite(IMAGE_FETCH_TIMEOUT_MS) && IMAGE_FETCH_TIMEOUT_MS > 0
      ? AbortSignal.timeout(IMAGE_FETCH_TIMEOUT_MS)
      : undefined;

  let response;
  try {
    response = await fetch(imageUrlString, {
      signal,
      redirect: 'follow',
      headers: { Accept: 'image/*,*/*' },
    });
  } catch (error) {
    throw Object.assign(new Error('Failed to download image'), { status: 502, cause: error });
  }

  if (!response.ok) {
    throw Object.assign(new Error(`Image URL returned ${response.status}`), { status: 502 });
  }

  const blob = await response.blob();
  const { RawImage } = await getTransformers();
  return RawImage.fromBlob(blob);
}

let segmenterPromise = null;

async function getSegmenter() {
  if (!segmenterPromise) {
    segmenterPromise = (async () => {
      const memProfile = getTransformersMemoryProfile();
      console.log(`--- Initializing background-removal pipeline (${RMBG_MODEL}, dtype=${RMBG_DTYPE}) ---`);
      if (memProfile === 'low') {
        console.log(
          '--- Memory profile: low (sequential ONNX, single-threaded ops; set TRANSFORMERS_MEMORY_PROFILE=high to disable) ---',
        );
      }
      const { pipeline } = await getTransformers();
      const config = await getConfigForModel(RMBG_MODEL, { userAgent: 'background-remover-server' });
      const session_options = getOnnxSessionOptionsForPipeline();
      const pipelineOptions = { dtype: RMBG_DTYPE, session_options };
      if (config) pipelineOptions.config = config;
      return pipeline('background-removal', RMBG_MODEL, pipelineOptions);
    })();
  }
  return segmenterPromise;
}

function sendJsonError(res, status, message) {
  res.status(status).type('application/json').send({ error: message });
}

async function removeBackgroundAndToPngBuffer(imageUrlString) {
  const image = await loadImageFromRemoteUrl(imageUrlString);
  const inputImage = await maybeResizeForSpeed(image, RMBG_MAX_SIDE);
  const segmenter = await getSegmenter();
  const results = await segmenter(inputImage);
  const resultImage = Array.isArray(results) ? results[0] : results;
  return resultImage.toSharp().png().toBuffer();
}

async function handleRemove(res, imageUrl) {
  try {
    const png = await removeBackgroundAndToPngBuffer(imageUrl);
    res.setHeader('Content-Type', 'image/png');
    res.setHeader('Content-Disposition', 'inline; filename="no-bg.png"');
    res.send(png);
  } catch (error) {
    const status = typeof error.status === 'number' ? error.status : 500;
    const message =
      status === 500 ? 'Background removal failed' : error.message || 'Request failed';
    if (status === 500) {
      console.error(error);
    }
    sendJsonError(res, status, message);
  }
}

const app = express();

app.post('/remove', express.json({ limit: '64kb' }), async (req, res) => {
  const url = typeof req.body?.url === 'string' ? req.body.url.trim() : '';
  if (!url) {
    sendJsonError(res, 400, 'Missing or invalid "url" in JSON body');
    return;
  }
  await handleRemove(res, url);
});

app.get('/remove', async (req, res) => {
  const url = typeof req.query.url === 'string' ? req.query.url.trim() : '';
  if (!url) {
    sendJsonError(res, 400, 'Missing or invalid "url" query parameter');
    return;
  }
  await handleRemove(res, url);
});

const server = createServer(app);

server.on('error', (err) => {
  console.error('HTTP server failed to start:', err.message);
  if (err.code === 'EADDRINUSE') {
    console.error(
      `Port ${PORT} is already in use. Set PORT in the environment or stop the other process (e.g. lsof -i :${PORT}).`,
    );
  }
  process.exit(1);
});

server.listen(PORT, HOST, () => {
  const addr = server.address();
  const where =
    typeof addr === 'object' && addr !== null
      ? `${addr.address === '::' ? '[::]' : addr.address}:${addr.port}`
      : `${HOST}:${PORT}`;
  console.log(`Listening (pid ${process.pid}) on http://${where}`);
  console.log('POST /remove with JSON {"url":"..."} or GET /remove?url=...');
  console.log(`Model: ${RMBG_MODEL} (set RMBG_MODEL to override)`);
});

function shutdown(signal) {
  console.log(`\n${signal} received, closing server…`);
  server.close(() => {
    console.log('Server closed.');
    process.exit(0);
  });
  setTimeout(() => process.exit(1), 10_000).unref();
}

process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));
