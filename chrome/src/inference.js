import * as ort from 'onnxruntime-web';

const IMAGE_SIZE = 384;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const REAL_THRESHOLD = 0.7;
const FAKE_THRESHOLD = 0.7;

let _session = null;

/**
 * Initialize the ONNX session from the model bundled with the extension.
 */
export async function initModel() {
  if (_session) return;

  ort.env.wasm.numThreads = 1;

  // Point onnxruntime-web to the WASM files bundled with the extension
  if (typeof chrome !== 'undefined' && chrome.runtime && chrome.runtime.getURL) {
    ort.env.wasm.wasmPaths = chrome.runtime.getURL('/');
  }

  // Load model from extension bundle
  const modelUrl = chrome.runtime.getURL('model.onnx');
  const response = await fetch(modelUrl);
  const modelBuffer = await response.arrayBuffer();

  _session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: ['wasm'],
  });
}

export function isModelReady() {
  return _session !== null;
}

/**
 * Preprocess an image into an ONNX tensor.
 * Replicates the Python preprocessing: resize 384x384, /255, ImageNet normalize, HWC->CHW.
 */
function preprocessImage(canvas, ctx) {
  ctx.drawImage(canvas._sourceImage, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const imageData = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE);
  const { data } = imageData;

  // Create Float32Array in [1, 3, 384, 384] layout (NCHW)
  const float32Data = new Float32Array(1 * 3 * IMAGE_SIZE * IMAGE_SIZE);

  for (let y = 0; y < IMAGE_SIZE; y++) {
    for (let x = 0; x < IMAGE_SIZE; x++) {
      const pixelIdx = (y * IMAGE_SIZE + x) * 4;
      const r = data[pixelIdx] / 255.0;
      const g = data[pixelIdx + 1] / 255.0;
      const b = data[pixelIdx + 2] / 255.0;

      // ImageNet normalization + CHW layout
      const pos = y * IMAGE_SIZE + x;
      float32Data[0 * IMAGE_SIZE * IMAGE_SIZE + pos] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      float32Data[1 * IMAGE_SIZE * IMAGE_SIZE + pos] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      float32Data[2 * IMAGE_SIZE * IMAGE_SIZE + pos] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
  }

  return new ort.Tensor('float32', float32Data, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

/**
 * Load an image source (URL, data URL, or Blob) into an HTMLImageElement.
 * @returns {Promise<HTMLImageElement>}
 */
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(new Error('Failed to load image'));
    if (src instanceof Blob) {
      img.src = URL.createObjectURL(src);
    } else {
      img.src = src;
    }
  });
}

/**
 * Draw bounding box + label on a canvas, replicating the Python draw_bboxes_numpy.
 * Returns a data URL of the annotated image.
 */
function drawBbox(img, label, color, confidence) {
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth || img.width;
  canvas.height = img.naturalHeight || img.height;
  const ctx = canvas.getContext('2d');

  ctx.drawImage(img, 0, 0);

  const w = canvas.width;
  const h = canvas.height;
  const bboxWidth = Math.max(8, Math.round(Math.min(w, h) * 0.01));
  const fontSize = Math.max(32, Math.round(Math.min(w, h) * 0.06));

  // Draw border rectangle
  ctx.strokeStyle = color;
  ctx.lineWidth = bboxWidth;
  ctx.strokeRect(bboxWidth / 2, bboxWidth / 2, w - bboxWidth, h - bboxWidth);

  // Draw label text
  const text = `${label} ${(confidence * 100).toFixed(1)}%`;
  ctx.font = `bold ${fontSize}px sans-serif`;
  ctx.textBaseline = 'top';
  const textMetrics = ctx.measureText(text);
  const textHeight = fontSize;
  const textX = 24;
  const textY = bboxWidth + 14;

  // Text background
  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  ctx.fillRect(textX - 4, textY - 4, textMetrics.width + 8, textHeight + 8);

  // Text
  ctx.fillStyle = color;
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 2;
  ctx.strokeText(text, textX, textY);
  ctx.fillText(text, textX, textY);

  return canvas.toDataURL('image/png');
}

/**
 * Run inference on an image source (URL, data URL, File, or Blob).
 * @param {string|Blob|File} imageSource
 * @returns {Promise<{label, confidence, real_confidence, fake_confidence, image}>}
 */
export async function runInference(imageSource) {
  if (!_session) {
    throw new Error('Model not initialized. Call initModel() first.');
  }

  const img = await loadImage(imageSource);

  // Create offscreen canvas for preprocessing
  const canvas = document.createElement('canvas');
  canvas.width = IMAGE_SIZE;
  canvas.height = IMAGE_SIZE;
  canvas._sourceImage = img;
  const ctx = canvas.getContext('2d');

  const inputTensor = preprocessImage(canvas, ctx);

  // Run inference
  const inputName = _session.inputNames[0];
  const feeds = { [inputName]: inputTensor };
  const results = await _session.run(feeds);

  // Get logits from the first output
  const outputNames = _session.outputNames;
  let logits;
  if (outputNames.includes('output')) {
    logits = results['output'].data;
  } else {
    logits = results[outputNames[0]].data;
  }

  // Softmax: logits[0] = fake, logits[1] = real
  const probs = softmax(Array.from(logits));
  const fakeConf = probs[0];
  const realConf = probs[1];

  // Thresholding (matches Python logic)
  let predIdx;
  if (realConf >= REAL_THRESHOLD) {
    predIdx = 1;
  } else if (fakeConf >= FAKE_THRESHOLD) {
    predIdx = 0;
  } else {
    predIdx = fakeConf > realConf ? 0 : 1;
  }

  const classNames = ['fake', 'real'];
  const predLabel = classNames[predIdx];
  const confidence = probs[predIdx];
  const color = predIdx === 0 ? '#f44336' : '#4CAF50';
  const displayLabel = predLabel.charAt(0).toUpperCase() + predLabel.slice(1);

  const annotatedImage = drawBbox(img, displayLabel, color, confidence);

  return {
    label: predLabel,
    confidence,
    real_confidence: realConf,
    fake_confidence: fakeConf,
    image: annotatedImage,
  };
}
