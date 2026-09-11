export type Verdict = {
  label: "fake" | "real";
  /** Probabilité softmax de la classe « générée par IA ». */
  fake: number;
  /** Probabilité softmax de la classe « réelle ». */
  real: number;
};

export type Subject = {
  id: string;
  title: string;
  src: string;
  source: "example" | "upload";
  /** Libellé du générateur (exemples) ou du format (upload). */
  detail: string;
  file?: File;
};

export type Analysis =
  | { status: "pending" }
  | { status: "done"; verdict: Verdict; elapsed: number; live: boolean }
  | { status: "error"; message: string };
