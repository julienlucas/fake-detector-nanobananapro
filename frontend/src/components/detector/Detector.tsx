import { useEffect, useRef, useState } from "react";
import { Clock, ImageUp, RotateCcw, ScanSearch, ShieldAlert, ShieldCheck, TriangleAlert } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { api } from "@/api";
import { useElapsed } from "@/hooks/useTimer";
import { ImagePanel, checkImage } from "./ImagePanel";
import { DOUBT_THRESHOLD, exampleImages, type ExampleImage } from "./examples";
import { pipelineSteps } from "./pipeline";
import type { Analysis, Subject, Verdict } from "./types";

const DEFAULT_EXAMPLE = exampleImages[0];

const pct = (v: number) =>
  `${(v * 100).toLocaleString("fr-FR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })} %`;

function subjectFromExample(ex: ExampleImage): Subject {
  return { id: ex.id, title: ex.title, src: ex.src, source: "example", detail: ex.source };
}

/** Barre de probabilité d'une classe : la classe retenue porte sa couleur de statut. */
function ProbBar({ label, value, tone, strong }: { label: string; value: number; tone: "fake" | "real"; strong: boolean }) {
  return (
    <div className="grid grid-cols-[4.5rem_1fr_3.75rem] items-center gap-3">
      <span className="text-sm font-medium">{label}</span>
      <span className="h-2.5 overflow-hidden rounded-r-[4px] bg-paper-3">
        <span
          className={cn(
            "block h-full rounded-r-[4px] transition-[width] duration-700",
            !strong ? "bg-sand-deep" : tone === "fake" ? "bg-destructive" : "bg-success",
          )}
          style={{ width: `${Math.max(value * 100, 0.5)}%` }}
        />
      </span>
      <span className="mono-xs text-right text-ink-muted">{pct(value)}</span>
    </div>
  );
}

function VerdictView({ verdict, elapsed, live }: { verdict: Verdict; elapsed: number; live: boolean }) {
  const fake = verdict.label === "fake";
  const confidence = fake ? verdict.fake : verdict.real;
  const doubtful = confidence < DOUBT_THRESHOLD;

  return (
    <div className="rise-in">
      <div className="flex flex-wrap items-center gap-2">
        {live ? (
          <span className="mono-xs inline-flex items-center gap-1 text-muted-foreground">
            <Clock className="size-3" /> analyse en direct ·{" "}
            {elapsed.toLocaleString("fr-FR", { maximumFractionDigits: 1 })} s
          </span>
        ) : (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="mono-xs cursor-help text-muted-foreground underline decoration-dotted underline-offset-4">
                résultat pré-calculé
              </span>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              Verdict du modèle de production (ONNX FP16) calculé hors ligne sur cette image. « Analyser en
              direct » interroge l'API.
            </TooltipContent>
          </Tooltip>
        )}
      </div>

      <div className={cn("mt-4 flex items-center gap-2", fake ? "text-destructive" : "text-success")}>
        {fake ? <ShieldAlert className="size-5" /> : <ShieldCheck className="size-5" />}
        <span className="display-sm">{fake ? "Générée par IA" : "Image réelle"}</span>
      </div>
      <div className="font-display mt-3 text-6xl font-normal tracking-tight tabular-nums">{pct(confidence)}</div>
      <div className="mt-1 text-sm text-ink-muted">de confiance dans ce verdict</div>

      <div className="mt-6 space-y-2.5">
        <ProbBar label="IA" value={verdict.fake} tone="fake" strong={fake} />
        <ProbBar label="Réelle" value={verdict.real} tone="real" strong={!fake} />
      </div>

      {doubtful ? (
        <p className="mt-5 flex gap-2 rounded-sm border border-warning/40 bg-warning/10 px-3 py-2.5 text-xs leading-relaxed text-ink-muted">
          <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-warning" />
          <span>
            <strong className="font-medium text-ink">Zone de doute.</strong> Aucune classe ne dépasse{" "}
            {Math.round(DOUBT_THRESHOLD * 100)} % : le modèle penche pour « {fake ? "IA" : "réelle"} » sans
            certitude. À lire comme une présomption, pas comme un verdict.
          </span>
        </p>
      ) : null}
    </div>
  );
}

export function Detector() {
  const [subject, setSubject] = useState<Subject>(() => subjectFromExample(DEFAULT_EXAMPLE));
  const [analysis, setAnalysis] = useState<Analysis>({
    status: "done",
    verdict: DEFAULT_EXAMPLE.precomputed,
    elapsed: 0,
    live: false,
  });
  const [stage, setStage] = useState(0);
  const [dragOver, setDragOver] = useState(false);
  const pending = analysis.status === "pending";
  const elapsed = useElapsed(pending);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);
  const token = useRef(0);
  const objectUrl = useRef<string | null>(null);

  const clearTimers = () => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
  };

  useEffect(
    () => () => {
      clearTimers();
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    },
    [],
  );

  const releaseUpload = () => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = null;
  };

  /* ---------- analyse ---------- */

  async function analyze(target: Subject) {
    const run = ++token.current;
    clearTimers();
    setStage(0);
    setAnalysis({ status: "pending" });

    // Animation estimée de la trace (le backend répond en un bloc).
    let acc = 0;
    pipelineSteps.forEach((s, i) => {
      if (s.ms > 0 && i + 1 < pipelineSteps.length) {
        acc += s.ms;
        timers.current.push(setTimeout(() => setStage(i + 1), acc));
      }
    });

    const started = performance.now();
    try {
      let file = target.file;
      if (!file) {
        const blob = await fetch(target.src).then((r) => r.blob());
        file = new File([blob], target.src.split("/").pop() || "image", { type: blob.type || "image/jpeg" });
      }
      const res = await api.inference(file);
      if (run !== token.current) return;
      setAnalysis({
        status: "done",
        verdict: { label: res.label, fake: res.fake_confidence, real: res.real_confidence },
        elapsed: Math.round((performance.now() - started) / 100) / 10,
        live: true,
      });
    } catch (err) {
      if (run !== token.current) return;
      const message = err instanceof Error ? err.message : "Erreur de connexion au backend";
      setAnalysis({ status: "error", message });
      toast.error("L'analyse a échoué", { description: message });
    } finally {
      if (run === token.current) clearTimers();
    }
  }

  const selectExample = (id: string) => {
    const ex = exampleImages.find((e) => e.id === id);
    if (!ex || ex.id === subject.id) return;
    token.current++;
    clearTimers();
    releaseUpload();
    setSubject(subjectFromExample(ex));
    setAnalysis({ status: "done", verdict: ex.precomputed, elapsed: 0, live: false });
  };

  const upload = (file: File) => {
    releaseUpload();
    const src = URL.createObjectURL(file);
    objectUrl.current = src;
    const next: Subject = {
      id: `upload-${Date.now()}`,
      title: file.name,
      src,
      source: "upload",
      detail: (file.type.split("/")[1] ?? "image").toUpperCase(),
      file,
    };
    setSubject(next);
    analyze(next);
  };

  const removeUpload = () => {
    token.current++;
    clearTimers();
    releaseUpload();
    setSubject(subjectFromExample(DEFAULT_EXAMPLE));
    setAnalysis({ status: "done", verdict: DEFAULT_EXAMPLE.precomputed, elapsed: 0, live: false });
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    if (pending) return;
    const file = e.dataTransfer.files?.[0];
    if (!file) return;
    const error = checkImage(file);
    if (error) {
      toast.error("Image refusée", { description: error });
      return;
    }
    upload(file);
  };

  const verdict = analysis.status === "done" ? analysis.verdict : null;
  const fake = verdict?.label === "fake";

  return (
    <div>
      <ImagePanel
        active={subject}
        onSelectExample={selectExample}
        onUpload={upload}
        onRemoveUpload={removeUpload}
        busy={pending}
      />


      <h3 className="display-sm pb-3 pt-8">Le verdict</h3>

      <div className="card-paper overflow-hidden bg-white">
        <div className="grid md:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
          {/* image analysée — zone de dépôt */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              if (!pending) setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className="relative flex min-h-[18rem] items-center justify-center border-b border-hairline bg-paper-2 p-4 sm:min-h-[26rem] sm:p-6 sm:pt-10 md:border-b-0 md:border-r"
          >
            <div className="relative inline-flex max-w-full">
              <img
                key={subject.src}
                src={subject.src}
                alt={subject.title}
                className="rise-in block h-auto max-h-[16rem] w-auto max-w-full rounded-md object-contain sm:max-h-[23rem]"
              />
              {verdict ? (
                <>
                  <span
                    aria-hidden
                    className={cn(
                      "pointer-events-none absolute inset-0 rounded-md ring-2 ring-inset",
                      fake ? "ring-destructive" : "ring-success",
                    )}
                  />
                  <span
                    className={cn(
                      "mono-xs absolute bottom-2 left-2 rounded-sm px-2 py-1 text-on-ink",
                      fake ? "bg-destructive" : "bg-success",
                    )}
                  >
                    {fake ? "IA" : "Réelle"} · {pct(fake ? verdict.fake : verdict.real)}
                  </span>
                </>
              ) : null}
              {pending ? (
                <span aria-hidden className="absolute inset-0 overflow-hidden rounded-md">
                  <span className="scan-line absolute inset-x-0 h-16 bg-gradient-to-b from-transparent via-gold-300/45 to-transparent" />
                </span>
              ) : null}
            </div>

            <span className="mono-xs absolute left-4 top-3 hidden text-ink-faint sm:block">
              glissez-déposez une image ici
            </span>

            {dragOver ? (
              <div className="absolute inset-2 grid place-items-center rounded-md border-2 border-dashed border-brand bg-brand-surface-strong/90">
                <span className="flex items-center gap-2 text-sm font-medium">
                  <ImageUp className="size-4 text-brand-deep" /> Déposez l'image pour l'analyser
                </span>
              </div>
            ) : null}
          </div>

          {/* verdict */}
          <div className="flex min-h-[22rem] flex-col p-5 sm:p-6">
            <div className="flex items-center justify-between gap-3">
              <span className="eyebrow">Verdict</span>
              <span className="mono-xs max-w-[60%] truncate text-ink-faint" title={subject.title}>
                {subject.title}
              </span>
            </div>

            <div className="mt-4 flex-1">
              {analysis.status === "done" ? (
                <VerdictView verdict={analysis.verdict} elapsed={analysis.elapsed} live={analysis.live} />
              ) : null}

              {analysis.status === "pending" ? (
                <div className="rise-in space-y-2 rounded-2xl border border-dashed border-brand/50 bg-brand-surface/40 px-4 py-3.5">
                  {pipelineSteps.map((s, i) => {
                    const done = stage > i;
                    const running = stage === i;
                    return (
                      <div
                        key={s.key}
                        className={cn("flex items-center gap-2.5 text-xs transition-opacity", !done && !running && "opacity-40")}
                      >
                        {done ? (
                          <span className="size-3.5 shrink-0 rounded-full bg-success/80" />
                        ) : (
                          <span
                            className={cn("size-3.5 shrink-0 rounded-full border border-brand", running && "signal-dot bg-brand")}
                          />
                        )}
                        <span className={cn("font-medium", done && "text-ink-muted")}>{s.label}</span>
                        {running ? <span className="mono-xs text-muted-foreground">en cours…</span> : null}
                      </div>
                    );
                  })}
                  <p className="mono-xs pt-1 text-muted-foreground">
                    {elapsed}s · ~0,4 s à chaud, plus long au premier appel (le modèle se charge)
                  </p>
                </div>
              ) : null}

              {analysis.status === "error" ? (
                <p className="rise-in flex gap-2 rounded-2xl border border-destructive/40 bg-destructive/5 px-4 py-3.5 text-sm text-destructive">
                  <ShieldAlert className="mt-0.5 size-4 shrink-0" />
                  {analysis.message}
                </p>
              ) : null}

            </div>

            <div className="mt-6 flex flex-wrap items-center gap-3 border-t border-hairline pt-4">
              <Button variant="brand" onClick={() => analyze(subject)} disabled={pending}>
                {analysis.status === "done" && analysis.live ? <RotateCcw /> : null}
                {analysis.status === "done" && analysis.live ? "Relancer l'analyse" : "Analyser en direct"}
                <ScanSearch />
              </Button>
              <span className="mono-xs text-muted-foreground">
                sortie softmax · seuil de doute {Math.round(DOUBT_THRESHOLD * 100)} %
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
