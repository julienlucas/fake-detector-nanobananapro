import { build } from "esbuild";
import { cpSync, mkdirSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const distDir = join(__dirname, "dist");
const modelSrc = join(__dirname, "model.onnx");

mkdirSync(distDir, { recursive: true });

if (!existsSync(modelSrc)) {
  throw new Error("chrome/model.onnx introuvable. Placez le modèle dans chrome/model.onnx");
}
cpSync(modelSrc, join(distDir, "model.onnx"));
console.log("Modèle copié dans dist.");

// Find onnxruntime-web WASM files
const ortDir = dirname(require.resolve("onnxruntime-web"));

// Bundle popup.js
await build({
  entryPoints: [join(__dirname, "src/popup-entry.js")],
  bundle: true,
  outfile: join(distDir, "popup.js"),
  format: "iife",
  target: "chrome110",
  minify: false,
});

// Bundle content.js
await build({
  entryPoints: [join(__dirname, "src/content-entry.js")],
  bundle: true,
  outfile: join(distDir, "content.js"),
  format: "iife",
  target: "chrome110",
  minify: false,
});

// Copy WASM files from onnxruntime-web to dist
const wasmFiles = [
  "ort-wasm-simd-threaded.wasm",
  "ort-wasm-simd.wasm",
  "ort-wasm.wasm",
  "ort-wasm-simd-threaded.jsep.wasm",
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
  "popup.html",
  "styles.css",
  "content.css",
  "background.js",
  "manifest.json",
];

for (const file of staticFiles) {
  cpSync(join(__dirname, file), join(distDir, file));
}

// Copy icons
if (existsSync(join(__dirname, "icons"))) {
  cpSync(join(__dirname, "icons"), join(distDir, "icons"), { recursive: true });
}

// Copy images
if (existsSync(join(__dirname, "images"))) {
  cpSync(join(__dirname, "images"), join(distDir, "images"), {
    recursive: true,
  });
}

console.log("Build complete! Load chrome/dist as unpacked extension.");
