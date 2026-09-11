const API_URL = import.meta.env.VITE_RAILWAY_API_URL || "";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function parse<T>(response: Response): Promise<T> {
  let data: any = null;
  try {
    data = await response.json();
  } catch {
    /* corps vide */
  }
  if (!response.ok || data?.error) {
    throw new ApiError(data?.error || `Erreur serveur (${response.status})`, response.status);
  }
  return data as T;
}

/** Réponse de `backend/views.py` : probabilités softmax des deux classes. */
export type InferenceResponse = {
  label: "fake" | "real";
  confidence: number;
  real_confidence: number;
  fake_confidence: number;
  /** L'image annotée (cadre + étiquette) en data URL PNG. */
  image?: string;
};

export const api = {
  /** Envoie l'image au modèle ONNX (EfficientNetV2-S, FP16). */
  inference(file: File) {
    const body = new FormData();
    body.append("file", file);
    return fetch(`${API_URL}/api/inference`, { method: "POST", body }).then((r) =>
      parse<InferenceResponse>(r),
    );
  },
};
