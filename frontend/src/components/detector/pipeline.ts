/**
 * Étapes de l'inférence telles qu'implémentées dans `backend/inference_onnx.py`.
 * Les durées servent uniquement à animer la trace pendant l'attente : le backend
 * répond en un bloc, le verdict réel arrive avec la réponse.
 */
export type PipelineStep = {
  key: "upload" | "resize" | "backbone" | "head";
  label: string;
  ms: number;
};

export const pipelineSteps: PipelineStep[] = [
  { key: "upload", label: "Envoi de l'image", ms: 350 },
  { key: "resize", label: "Redimension 384 × 384 et normalisation ImageNet", ms: 250 },
  { key: "backbone", label: "EfficientNetV2-S — extraction des features", ms: 500 },
  { key: "head", label: "Double pooling Avg + Max, puis tête de classification", ms: 0 },
];
