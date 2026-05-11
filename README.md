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
  Runs [`src/image.js`](src/image.js). Default **`RMBG_MODEL`** is `briaai/RMBG-2.0` (gated); override in `.env` or the shell.
- `npm run imageFreeLicence`  
  Runs [`src/imageFreeLicence.js`](src/imageFreeLicence.js). Default **`RMBG_MODEL`** is `Xenova/modnet` (Apache 2.0); override for other supported models (for example U-2-Net below).
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
| `RMBG_MODEL` | `Xenova/modnet` (server) / see CLI defaults | Model id: standard **`background-removal`** models, or **`BritishWerewolf/U-2-Net`** (custom path; Apache 2.0, see below) |
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

   Another Apache 2.0 option that is often lighter than RMBG-2.0:

   ```env
   RMBG_MODEL=BritishWerewolf/U-2-Net
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

### U-2-Net ([BritishWerewolf/U-2-Net](https://huggingface.co/BritishWerewolf/U-2-Net))

This repo supports the Apache 2.0 ONNX build **BritishWerewolf/U-2-Net** via **`RMBG_MODEL=BritishWerewolf/U-2-Net`**. The Hub repo only publishes **`onnx/model.onnx` (fp32)** — there is no `model_quantized.onnx`, so **`RMBG_DTYPE=q8` (and other dtypes) are ignored** for this model with a console warning; loading always uses fp32.

Transformers.js does not register `U2NetImageProcessor` yet. **By default, preprocessing matches [rembg’s `BaseSession.normalize`](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/base.py)** (RGB → exact **320×320 stretch** with LANCZOS, divide by **global max**, then ImageNet mean/std). That is what most public `u2net.onnx` builds expect; the [official U-2-Net paper repo](https://github.com/xuebinqin/U-2-Net) also recommends **320×320** for the pretrained SOD weights. Letterboxing (Hub [`preprocessor_config.json`](https://huggingface.co/BritishWerewolf/U-2-Net/blob/main/preprocessor_config.json)) misaligns the saliency map when the mask is resized back to the original image, which looks like a bad cutout or a “cropped” foreground. To use Hub-style letterbox instead, set **`U2NET_PREPROCESS=hf`**.

Inference uses **`AutoModel`** and the composite ONNX output described in the model `config.json`. The library would otherwise print `Unknown model class "u2net"` — that path is normal and is filtered in this project in favor of a short explanatory log line.

Mask post-processing follows the same idea as [rembg’s `U2netSession`](https://github.com/danielgatis/rembg/blob/main/rembg/sessions/u2net.py): optional sigmoid on logits, then **per-image min–max normalization** so the saliency map uses the full 0–1 range before converting to 8-bit alpha (otherwise the mask often stays mid-gray and the cutout looks blurred or washed out). Set **`U2NET_SKIP_MASK_MINMAX=1`** only if you need the raw ONNX activations as alpha.

Other U-2-Net repos on the Hub are not wired unless added to [`U2NET_SUPPORTED_MODEL_IDS`](src/lib/u2net-infer.js).

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
