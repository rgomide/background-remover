export async function maybeResizeForSpeed(image, maxSide) {
  if (!Number.isInteger(maxSide) || maxSide <= 0) {
    return image;
  }
  const [width, height] = image.size;
  const currentMaxSide = Math.max(width, height);
  if (currentMaxSide <= maxSide) {
    return image;
  }
  const scale = maxSide / currentMaxSide;
  const resizedWidth = Math.max(1, Math.round(width * scale));
  const resizedHeight = Math.max(1, Math.round(height * scale));
  console.log(`Resizing ${width}x${height} -> ${resizedWidth}x${resizedHeight} for faster inference`);
  return image.resize(resizedWidth, resizedHeight);
}
