import { useState, type CSSProperties } from "react";
import { Table2 } from "lucide-react";
import { Section } from "@/components/site/primitives";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/*
 * Précision (accuracy) sur le split test du dataset Hugging Face (2 000 images), relevée
 * dans les noms des checkpoints publiés sur huggingface.co/julienlucas/fakefinder.
 * Le chiffre « selfies » vient du README : mesuré hors dataset, sur des photos personnelles.
 */
const levels = [
  {
    label: "Le départ",
    value: "86,7 %",
    unit: "de précision",
    detail: "MobileNetV3 · 5,4 M paramètres",
    setup: "Transfer learning ImageNet sur Mac M1, tête puis dégel complet, focal loss et échantillonnage pondéré.",
    tone: "muted" as const,
  },
  {
    label: "Le modèle final",
    value: "91,5 %",
    unit: "de précision",
    detail: "EfficientNetV2-S · F1 macro 91,5 %",
    setup: "384 px · Optuna (20 essais) · double pooling Avg + Max · focal γ = 3 · augmentations anti-capteur · SWA · ONNX FP16",
    tone: "brand" as const,
  },
  {
    label: "Selfies smartphone",
    value: "80 %",
    unit: "de précision",
    detail: "le cas le plus dur, hors dataset",
    setup: "Des photos réelles dont le logiciel du téléphone lisse la peau : le piège exact qui a fait écarter le run à 92 %.",
    tone: "ref" as const,
  },
];

type Row = {
  stage: string;
  params: string;
  res: string;
  acc: number;
  hint: string;
  highlight?: boolean;
  discarded?: boolean;
};

const rows: Row[] = [
  { stage: "MobileNetV3-Large", params: "5,4 M", res: "256 px", acc: 86.7, hint: "fine-tuning complet, int8" },
  { stage: "ResNet18", params: "11,7 M", res: "224 px", acc: 87.4, hint: "trial Optuna #8 (86,3 %) puis fine-tuning" },
  { stage: "EfficientNetV2-S", params: "21,5 M", res: "384 px", acc: 90.8, hint: "trial Optuna #13" },
  { stage: "Run à 92 %", params: "21,5 M", res: "384 px", acc: 92, hint: "écarté : a mémorisé le capteur", discarded: true },
  {
    stage: "« Universal Real »",
    params: "21,5 M",
    res: "384 px",
    acc: 91.5,
    hint: "modèle final · F1 macro 91,5 %",
    highlight: true,
  },
];

/* Axe tronqué à 80–95 % : un nuage de points n'a pas besoin de la ligne de zéro, une barre si. */
const AXIS_MIN = 80;
const AXIS_MAX = 95;
const ticks = [80, 85, 90, 95];
const x = (v: number) => `${((v - AXIS_MIN) / (AXIS_MAX - AXIS_MIN)) * 100}%`;
const fr = (v: number) => v.toLocaleString("fr-FR", { maximumFractionDigits: 1 });

function ProgressChart() {
  const [table, setTable] = useState(false);
  return (
    <figure className="card-paper border-hairline p-6 sm:p-8">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 max-w-2xl">
          <figcaption className="display-sm">De 86,7 à 91,5 % de précision</figcaption>
          <p className="mono-xs mt-1 text-muted-foreground">
            précision sur le split test · 2 000 images · axe de 80 à 95 %
          </p>
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="-mt-1 ml-auto shrink-0"
          onClick={() => setTable((t) => !t)}
          aria-pressed={table}
        >
          <Table2 /> {table ? "Graphique" : "Tableau"}
        </Button>
      </div>

      {table ? (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-muted-foreground">
                <th className="py-2 pr-4 font-medium">Étape</th>
                <th className="py-2 pr-4 font-medium">Paramètres</th>
                <th className="py-2 pr-4 font-medium">Résolution</th>
                <th className="py-2 pr-4 font-medium">Précision</th>
                <th className="py-2 font-medium">Note</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.stage} className="border-t border-hairline">
                  <td className="py-2 pr-4 font-medium">{r.stage}</td>
                  <td className="py-2 pr-4">{r.params}</td>
                  <td className="py-2 pr-4">{r.res}</td>
                  <td className={cn("py-2 pr-4", r.discarded && "text-ink-faint line-through")}>{fr(r.acc)} %</td>
                  <td className="py-2 text-ink-muted">{r.hint}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="mt-6">
          {/* axe */}
          <div className="grid gap-1 sm:grid-cols-[12rem_1fr]">
            <span />
            <div className="relative h-5">
              {ticks.map((t) => (
                <span
                  key={t}
                  className="mono-xs absolute -translate-x-1/2 whitespace-nowrap text-ink-faint"
                  style={{ left: x(t) }}
                >
                  {t} %
                </span>
              ))}
            </div>
          </div>
          <div className="mt-1 space-y-1">
            {rows.map((r) => (
              <div key={r.stage} className="grid items-center gap-1 sm:grid-cols-[12rem_1fr]">
                <div className="py-1">
                  <span className={cn("block text-sm font-medium", r.discarded && "text-ink-faint")}>{r.stage}</span>
                  <span className="block text-[0.7rem] leading-snug text-muted-foreground">
                    {r.params} · {r.res}
                  </span>
                </div>
                <div className="relative h-9">
                  {/* grille */}
                  {ticks.map((t) => (
                    <span
                      key={t}
                      aria-hidden
                      className="absolute inset-y-0 w-px bg-hairline"
                      style={{ left: x(t) }}
                    />
                  ))}
                  {/* tige depuis le bord de l'axe */}
                  <span
                    aria-hidden
                    className={cn(
                      "absolute top-1/2 h-px -translate-y-1/2",
                      r.discarded ? "border-t border-dashed border-sand-deep" : "bg-sand-deep",
                    )}
                    style={{ left: 0, width: x(r.acc) }}
                  />
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span
                        className="absolute top-1/2 grid size-7 -translate-x-1/2 -translate-y-1/2 cursor-default place-items-center"
                        style={{ left: x(r.acc) }}
                      >
                        <span
                          className={cn(
                            "size-3 rounded-full ring-2 ring-paper",
                            r.highlight && "size-3.5 bg-chart-after",
                            r.discarded && "border-2 border-dashed border-sand-deep bg-paper",
                            !r.highlight && !r.discarded && "bg-chart-before",
                          )}
                        />
                      </span>
                    </TooltipTrigger>
                    <TooltipContent>
                      {r.stage} · {fr(r.acc)} % — {r.hint}
                    </TooltipContent>
                  </Tooltip>
                  <span
                    className={cn(
                      "mono-xs absolute top-1/2 -translate-y-1/2 pl-4 tabular-nums",
                      r.highlight ? "font-medium text-ink" : "text-ink-muted",
                      r.discarded && "text-ink-faint line-through",
                    )}
                    style={{ left: x(r.acc) }}
                  >
                    {fr(r.acc)} %
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </figure>
  );
}

const levers = [
  {
    title: "Un dataset Nano Banana Pro constitué par scraping",
    text: "Peu d'images Nano Banana Pro existaient au lancement du modèle : ~3 000 ont été scrapées, puis combinées à ~3 000 images Midjourney, DALL-E et Stable Diffusion et ~6 000 photos réelles. 12 695 images au total, publiées sur Hugging Face.",
  },
  {
    title: "Un backbone EfficientNetV2-S à 384 px",
    text: "+3,4 points face au meilleur ResNet18 : le plus gros gain mesuré du projet, obtenu en changeant d'architecture et de résolution plutôt qu'en grossissant le modèle.",
  },
  {
    title: "Des augmentations qui effacent le capteur",
    text: "JPEG agressif (qualité 20 à 80, 60 % des images), flou et netteté, autocontraste, perspective, bruit capteur : une vraie photo ne se reconnaît plus à sa signature, seul l'artefact de génération reste exploitable.",
  },
  {
    title: "Optuna, TPE et Hyperband",
    text: "Vingt essais sur neuf hyperparamètres, les plus faibles coupés tôt par Hyperband. Le trial #13 (90,8 %) sert de base saine à tout le reste.",
  },
  {
    title: "Double pooling Avg + Max",
    text: "La moyenne capte l'allure générale de l'image, le maximum l'artefact local — le pixel trop parfait ou trop bruité. Les deux vecteurs sont concaténés avant la tête de classification.",
  },
  {
    title: "Focal loss",
    text: "γ = 2 puis 3 : la perte se concentre sur les exemples difficiles — les fakes subtils — plutôt que sur ceux que le modèle réussit déjà.",
  },
  {
    title: "Dégel progressif, learning rates différenciés",
    text: "La tête s'entraîne seule, puis les derniers blocs du backbone sont dégelés avec un learning rate 10 à 20 fois plus faible : on adapte les features ImageNet sans les détruire.",
  },
  {
    title: "SWA et early stopping sur le F1",
    text: "La moyenne des poids des dernières epochs lisse les derniers pourcents ; l'arrêt se décide sur le F1 macro, pour ne jamais récompenser un modèle qui favorise une classe.",
  },
];

export function Results() {
  return (
    <Section
      id="resultats"
      index="02"
      eyebrow="L'évaluation"
      title={
        <>
          91,5 % de précision et de F1 sur{" "}
          <span className="accent-italic">2 000 images</span> de test.
        </>
      }
      intro="Le split test du dataset : des images jamais vues pendant l'entraînement, réelles ou générées par Nano Banana Pro, Midjourney, DALL-E et Stable Diffusion. La précision dit combien d'images sont bien classées ; le F1 macro vérifie que ce score ne cache pas une classe sacrifiée."
    >
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {levels.map((l, i) => (
          <div
            key={l.label}
            className={cn(
              "flex h-full flex-col rounded-xl border p-8",
              l.tone === "muted" && "border-hairline bg-chart-after/5",
              l.tone === "brand" && "bg-chart-after/20 shadow-xl",
              l.tone === "ref" && "border-hairline bg-chart-ref/5",
            )}
          >
            <div className="flex items-center justify-between">
              <span className="eyebrow">{l.label}</span>
              <span className="mono-xs text-ink-faint">0{i + 1}</span>
            </div>
            <div className="font-display mt-4 text-5xl font-normal tracking-tight">{l.value}</div>
            <div className="mt-1 text-sm font-medium">{l.unit}</div>
            <div className="mono-xs mt-1 text-muted-foreground">{l.detail}</div>
            <p className="mt-4 border-t border-hairline pt-3 text-xs leading-relaxed text-ink-muted">{l.setup}</p>
          </div>
        ))}
      </div>

      <div className="mt-14">
        <ProgressChart />
      </div>

      {/* Ce que les chiffres autorisent à dire — et pas plus. Hors de la carte, exprès. */}
      <figure className="grid max-w-4xl gap-x-6 pt-18 sm:grid-cols-[3.5rem_1fr]">
        <span aria-hidden className="display-xl -mt-3 hidden select-none leading-none text-brand sm:block">
          &ldquo;
        </span>
        <div>
          <span className="eyebrow">Ce que les chiffres disent</span>
          <blockquote className="display-md mt-3 text-ink">
            Sur <strong className="font-semibold text-ink">2 000 images</strong> jamais vues, le modèle en
            classe correctement <span className="accent-italic">91,5 %</span>, avec un F1 macro identique.
            Le run à 92 % a été écarté : il avait appris le capteur, pas la génération. Sur des selfies
            smartphone, le cas le plus dur, la précision tombe à{" "}
            <span className="accent-italic">80 %</span> — c'est là que se joue la suite.
          </blockquote>
        </div>
      </figure>

      <div className="mt-16">
        <h3 className="display-md max-w-3xl">Ce qui a été fait</h3>
        <ol className="mt-6 space-y-1">
          {levers.map((l, i) => {
            // Du levier le plus structurant au plus fin : un or déjà léger au premier rang,
            // qui s'estompe jusqu'au papier au dernier.
            const strength = 0.2 - (i / (levers.length - 1)) * 0.185;
            return (
              <li
                key={l.title}
                data-strong={i === 0 || undefined}
                className="lever-row grid gap-1 rounded-sm px-4 py-4 sm:grid-cols-[2rem_16rem_1fr] sm:gap-4"
                style={{ "--lever-strength": `${Math.round(strength * 100)}%` } as CSSProperties}
              >
                <span className="lever-index mono-xs pt-1">0{i + 1}</span>
                <span className="text-sm font-medium">{l.title}</span>
                <span className="lever-text text-sm leading-relaxed">{l.text}</span>
              </li>
            );
          })}
        </ol>
      </div>
    </Section>
  );
}
