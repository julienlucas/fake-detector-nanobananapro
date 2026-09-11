import type { Verdict } from "./types";

export type ExampleImage = {
  id: string;
  src: string;
  title: string;
  source: string;
  /**
   * Verdict du modèle de production (ONNX FP16) sur cette image, pré-calculé en local
   * avec `backend/inference_onnx.py`. Affiché tel quel tant qu'aucune analyse en direct
   * n'a été lancée — et étiqueté comme tel.
   */
  precomputed: Verdict;
};

const verdict = (fake: number): Verdict => ({
  label: fake >= 0.5 ? "fake" : "real",
  fake,
  real: 1 - fake,
});

export const exampleImages: ExampleImage[] = [
  {
    id: "tea",
    src: "/static/abc85fe7-a3c2-4374-931a-4bee66bd4d9d_min.webp",
    title: "Thé royal",
    source: "Nano Banana Pro",
    precomputed: verdict(0.9967),
  },
  {
    id: "classroom",
    src: "/static/04efb32e-e9dc-4389-9b64-ad3af3e3389b_min.webp",
    title: "Salle de classe",
    source: "Nano Banana Pro",
    precomputed: verdict(0.9697),
  },
  {
    id: "party",
    src: "/static/2c857321-56eb-437d-9abb-f2f97e98628a_min.webp",
    title: "Soirée",
    source: "Nano Banana Pro",
    precomputed: verdict(0.9681),
  },
  {
    id: "beach",
    src: "/static/45180048-3fd8-441f-874f-c8a9b56b5b02_min.webp",
    title: "Plage, 1944",
    source: "Nano Banana Pro",
    precomputed: verdict(0.94),
  },
  {
    id: "selfie",
    src: "/static/G6GI9FqWMAAuZ_j.jpg",
    title: "Selfie de stars",
    source: "Générateur inconnu",
    precomputed: verdict(0.8627),
  },
  {
    id: "lake",
    src: "/static/GGQ_1723662752223_1723662762427.webp",
    title: "Bord de lac",
    source: "Générateur inconnu",
    precomputed: verdict(0.9363),
  },
  {
    id: "abbey-road",
    src: "/static/3a4cc2b2-7d83-4b2b-b365-0bf08b9d0c99_min.webp",
    title: "Abbey Road",
    source: "Image IA",
    precomputed: verdict(0.6444),
  },
];

export const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/webp"];
export const MAX_FILE_MB = 10;

/** Sous ce seuil, le backend donne quand même la classe majoritaire : on le signale. */
export const DOUBT_THRESHOLD = 0.7;
