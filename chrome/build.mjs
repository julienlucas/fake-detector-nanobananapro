import { build } from 'esbuild';
import { cpSync, mkdirSync, existsSync, createWriteStream } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import { pipeline } from 'stream/promises';

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const distDir = join(__dirname, 'dist');

const MODEL_URL = 'https://huggingface.co/julienlucas/fakefinder/resolve/main/models/best_f191.5_acc91.5_efficientnetv2s_fake_detector_fp16.onnx';
const MODEL_FILENAME = 'model.onnx';

mkdirSync(distDir, { recursive: true });

// Download model from HuggingFace if not already in dist/
const modelDest = join(distDir, MODEL_FILENAME);
if (!existsSync(modelDest)) {
  console.log('Downloading ONNX model from HuggingFace...');
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`Failed to download model: ${res.status}`);
  const fileStream = createWriteStream(modelDest);
  await pipeline(res.body, fileStream);
  console.log('Model downloaded.');
} else {
  console.log('Model already present, skipping download.');
}

// Find onnxruntime-web WASM files
const ortDir = dirname(require.resolve('onnxruntime-web'));

// Bundle popup.js
await build({
  entryPoints: [join(__dirname, 'src/popup-entry.js')],
  bundle: true,
  outfile: join(distDir, 'popup.js'),
  format: 'iife',
  target: 'chrome110',
  minify: false,
});

// Bundle content.js
await build({
  entryPoints: [join(__dirname, 'src/content-entry.js')],
  bundle: true,
  outfile: join(distDir, 'content.js'),
  format: 'iife',
  target: 'chrome110',
  minify: false,
});

// Copy WASM files from onnxruntime-web to dist
const wasmFiles = [
  'ort-wasm-simd-threaded.wasm',
  'ort-wasm-simd.wasm',
  'ort-wasm.wasm',
  'ort-wasm-simd-threaded.jsep.wasm',
];

for (const file of wasmFiles) {
  const src = join(ortDir, file);
  if (existsSync(src)) {
    cpSync(src, join(distDir, file));
    console.log(`Copied ${file}`);
  }
}

// Copy static files to dist
const staticFiles = [
  'popup.html',
  'popup.css',
  'content.css',
  'background.js',
  'manifest.json',
];

for (const file of staticFiles) {
  cpSync(join(__dirname, file), join(distDir, file));
}

// Copy icons
if (existsSync(join(__dirname, 'icons'))) {
  cpSync(join(__dirname, 'icons'), join(distDir, 'icons'), { recursive: true });
}

// Copy images
if (existsSync(join(__dirname, 'images'))) {
  cpSync(join(__dirname, 'images'), join(distDir, 'images'), { recursive: true });
}

console.log('Build complete! Load chrome/dist as unpacked extension.');
