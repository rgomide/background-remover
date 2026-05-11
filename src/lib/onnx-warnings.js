export function suppressOnnxShapeReuseWarnings() {
  const originalWrite = process.stderr.write.bind(process.stderr);
  process.stderr.write = (chunk, encoding, callback) => {
    const text = typeof chunk === 'string' ? chunk : chunk?.toString?.() ?? '';
    if (
      text.includes('AllocateMLValueTensorPreAllocateBuffer') &&
      text.includes('Shape mismatch attempting to re-use buffer')
    ) {
      if (typeof callback === 'function') callback();
      return true;
    }
    return originalWrite(chunk, encoding, callback);
  };
}
