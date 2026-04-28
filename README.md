# transformers-with-bert

Simple Node.js scripts for background removal using `@huggingface/transformers`.

## Requirements

- Node.js 18+ (recommended)
- npm

## Install

```bash
npm install
```

## Scripts

- `npm run image`  
  Runs `image.js` with `briaai/RMBG-2.0` (higher quality, gated on Hugging Face).
- `npm run imageFreeLicence`  
  Runs `imageFreeLicence.js` with `Xenova/modnet` (Apache 2.0, commercial use allowed).

## How to run

Both scripts accept one `<source>` argument, where source can be:

- A local image file path
- A local folder path (converts all supported images recursively)
- An image URL (`http`/`https`)

Examples:

```bash
npm run image -- "./photos/avatar.jpg"
npm run image -- "./photos"
npm run image -- "https://example.com/your-image.jpg"

npm run imageFreeLicence -- "./photos/avatar.jpg"
npm run imageFreeLicence -- "./photos"
npm run imageFreeLicence -- "https://example.com/your-image.jpg"
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`, `.tiff`, `.tif`.

Each processed input is saved in the project root as `<original_name>_no_bg*.png`.

## Hugging Face token (optional/required depending on model)

If you get `Unauthorized`:

1. Create a Read token at <https://huggingface.co/settings/tokens>
2. Add it to `.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

For `briaai/RMBG-2.0`, you may also need to accept the model license on:
<https://huggingface.co/briaai/RMBG-2.0>

## Speed up `image.js`

`image.js` (`briaai/RMBG-2.0`) is higher quality and usually slower than `imageFreeLicence.js` (`Xenova/modnet`).

You can speed it up with:

- `RMBG_DTYPE`: numeric precision (`fp32` default; try `fp16` or `q8` if supported on your setup)
- `RMBG_MAX_SIDE`: resizes input before inference (e.g. `1280`) for faster processing
- `PROCESS_PRIORITY`: sets process priority (-20 is the highest priority)

Examples:

```bash
RMBG_DTYPE=q8 npm run image -- "./photos"
RMBG_MAX_SIDE=1280 npm run image -- "./photos"
RMBG_DTYPE=q8 RMBG_MAX_SIDE=1280 npm run image -- "./photos"
RMBG_DTYPE=q8 RMBG_MAX_SIDE=1280 PROCESS_PRIORITY=-20 npm run image -- "./photos"
```
