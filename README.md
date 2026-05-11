# RMBG-2.0 and Modnet for background removal

Simple Node.js scripts for background removal using `@huggingface/transformers`.

Source lives under [`src/`](src/): CLI tools, the HTTP server, shared helpers in [`src/lib/`](src/lib/), and the optional sentiment demo in [`src/index.js`](src/index.js).

## Requirements

- Node.js 18+ (recommended)
- npm

## Install

```bash
npm install
```

## Scripts

- `npm run image`  
  Runs [`src/image.js`](src/image.js) with `briaai/RMBG-2.0` (higher quality, gated on Hugging Face).
- `npm run imageFreeLicence`  
  Runs [`src/imageFreeLicence.js`](src/imageFreeLicence.js) with `Xenova/modnet` (Apache 2.0, commercial use allowed).
- `npm run server` or `npm start`  
  Starts [`src/server.js`](src/server.js), an Express HTTP API that accepts a remote image URL and returns a PNG with the background removed.
- `npm run sentiment`  
  Runs the multilingual BERT sentiment demo in [`src/index.js`](src/index.js).

## HTTP server

Start the API (loads variables from `.env` if present):

```bash
npm run server
```

By default the server listens on **all interfaces** (`HOST=0.0.0.0`) and port **3000**. Override with `HOST` / `PORT`:

```bash
HOST=127.0.0.1 PORT=8080 npm run server
```

An empty value in `.env` such as `PORT=` is treated as “unset” (falls back to **3000**), not as an invalid port.

The background-removal model is chosen with **`RMBG_MODEL`** (default **`Xenova/modnet`**). The first request may be slow while the model downloads and initializes; later requests reuse the same pipeline.

If the process exits right away, the usual cause is **port already in use** (`EADDRINUSE`): another app (or a leftover `node src/server.js`) is bound to `PORT`. Run **`node --env-file=.env src/server.js`** and read the error, or try `PORT=3001 npm run server`. **Ctrl+C** triggers a clean shutdown (`SIGINT`).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/remove` | Primary API. JSON body: `{ "url": "<http or https image URL>" }`. |
| `GET` | `/remove` | Same behavior with query string: `?url=<encoded image URL>` (handy for quick tests; long URLs are easier with `POST`). |

**Success (200):** response body is **PNG** bytes (`Content-Type: image/png`, `Content-Disposition: inline; filename="no-bg.png"`).

**Errors:** JSON object `{ "error": "<message>" }` with an appropriate status code (for example **400** for a missing or disallowed URL, **502** if the image URL cannot be fetched).

Examples:

```bash
curl -o no-bg.png -X POST http://localhost:3000/remove \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/photo.jpg"}'

curl -o no-bg.png "http://localhost:3000/remove?url=https://t4.ftcdn.net/jpg/02/41/97/03/360_F_241970320_nMws6gsdbLZvcgblMuJOd7ewnbd7oBoS.jpg"
```

### Server environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PORT` | `3000` | HTTP listen port (invalid or empty `PORT=` falls back to 3000) |
| `HOST` | `0.0.0.0` | Bind address (`127.0.0.1` to accept local connections only) |
| `RMBG_MODEL` | `Xenova/modnet` | Hugging Face model id for `background-removal` |
| `RMBG_DTYPE` | `fp32` | Pipeline dtype (e.g. `fp16`, `q8` when supported) |
| `RMBG_MAX_SIDE` | `0` | If set to a positive integer, downscales the longest image side before inference |
| `IMAGE_FETCH_TIMEOUT_MS` | `30000` | Timeout for downloading the input image from `url` |
| `HF_TOKEN` / `HF_ACCESS_TOKEN` | — | Hugging Face token (needed for gated models and some downloads) |
| `PROCESS_PRIORITY` | unset | If set to an integer (e.g. `-10`), calls `os.setPriority` (often needs elevated privileges) |
| `RMBG_ALLOW_PRIVATE_URLS` | off | Set to `1` or `true` to allow fetching from private / loopback hosts (unsafe if the API is exposed beyond localhost) |
| `TRANSFORMERS_MEMORY_PROFILE` | auto | `auto`: use low-memory ONNX settings when `RENDER=true` or `LOW_MEMORY=1`. `low` / `high` force either mode. |
| `LOW_MEMORY` | off | Set to `1` or `true` to enable the same low-memory ONNX profile as Render without relying on `RENDER`. |
| `ORT_INTRA_OP_THREADS` | `1` | Used only in low-memory profile; ONNX intra-op thread count (≥ 1). |
| `ORT_INTER_OP_THREADS` | `1` | Used only in low-memory profile; ONNX inter-op thread count (≥ 1). |

**Security:** the server fetches arbitrary `http`/`https` URLs you pass in. By default it blocks common SSRF targets (for example `localhost`, link-local addresses, and hostnames that resolve to private IPs). Only enable `RMBG_ALLOW_PRIVATE_URLS` in controlled environments.

### Render.com and other ~512MB RAM hosts

Background-removal models are heavy in RAM. **`briaai/RMBG-2.0` in `fp32` often exceeds 512MB** during load or inference, so the process is killed.

1. **Use a smaller model** (strongly recommended on the free tier):

   ```env
   RMBG_MODEL=Xenova/modnet
   RMBG_DTYPE=fp32
   ```

2. **Cap input resolution** so activations stay smaller (try `640`–`1024` first):

   ```env
   RMBG_MAX_SIDE=768
   ```

3. **If the model offers a quantized dtype on your setup**, try `RMBG_DTYPE=q8` (only when the Hub ONNX build supports it).

4. **Low-memory ONNX profile** — when **`RENDER=true`** (Render sets this) or **`LOW_MEMORY=1`**, the server passes conservative `session_options` into Transformers.js (`sequential` execution, single-threaded intra/inter ops, CPU memory arena/pattern disabled). To turn that off on a larger instance while still on Render, set:

   ```env
   TRANSFORMERS_MEMORY_PROFILE=high
   ```

   Optional overrides: `ORT_INTRA_OP_THREADS`, `ORT_INTER_OP_THREADS` (defaults `1` in low profile).

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

Each processed input is saved under the **`out/`** directory in the **current working directory** (typically the repo root when you run `npm run image` from there) as `<name>_rmbg_no_bg*.png` or `<name>_modnet_no_bg*.png`.

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
