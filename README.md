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

Both scripts expect an image URL as argument:

```bash
npm run image -- "https://example.com/your-image.jpg"
npm run imageFreeLicence -- "https://example.com/your-image.jpg"
```

Each execution saves output as `resultado_<n>.png` in the project root.

## Hugging Face token (optional/required depending on model)

If you get `Unauthorized`:

1. Create a Read token at <https://huggingface.co/settings/tokens>
2. Add it to `.env`:

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

For `briaai/RMBG-2.0`, you may also need to accept the model license on:
<https://huggingface.co/briaai/RMBG-2.0>
