import fs from 'node:fs';
import path from 'node:path';

export const IMAGE_EXTENSIONS = new Set([
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
  '.bmp',
  '.gif',
  '.tiff',
  '.tif',
]);

export function isHttpUrl(value) {
  try {
    const parsed = new URL(value);
    return parsed.protocol === 'http:' || parsed.protocol === 'https:';
  } catch {
    return false;
  }
}

export async function collectImageFilesFromDirectory(directoryPath) {
  const entries = await fs.promises.readdir(directoryPath, { withFileTypes: true });
  const collected = [];

  for (const entry of entries) {
    const fullPath = path.join(directoryPath, entry.name);
    if (entry.isDirectory()) {
      collected.push(...(await collectImageFilesFromDirectory(fullPath)));
      continue;
    }

    if (entry.isFile() && IMAGE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      collected.push(fullPath);
    }
  }

  return collected;
}

export async function resolveSources(input) {
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

/** @param {{ fileTag: string }} options fileTag e.g. `rmbg_no_bg` or `modnet_no_bg` */
export function getUniqueOutputPath(source, executionTimeMs, outputDir, options) {
  const { fileTag } = options;
  const sourceName = isHttpUrl(source)
    ? 'resultado_url'
    : path.parse(source).name.replace(/[^a-zA-Z0-9-_]/g, '_');
  const baseName = `${sourceName}-execution_time_${executionTimeMs}ms`;

  let counter = 1;
  let candidate = path.join(outputDir, `${baseName}_${fileTag}.png`);
  while (fs.existsSync(candidate)) {
    candidate = path.join(outputDir, `${baseName}_${fileTag}_${counter}.png`);
    counter += 1;
  }
  return candidate;
}
