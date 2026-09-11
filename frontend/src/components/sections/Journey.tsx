import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/site/primitives";
import { cn } from "@/lib/utils";

/*
 * Chronologie reconstituée depuis training/*.py, l'historique git et les checkpoints publiés
 * sur huggingface.co/julienlucas/fakefinder (les scores sont dans les noms de fichiers :
 * best_86.7_mobilenetV3…, trial_8_resnet18…_0.8630, best_87.4_resnet18…,
 * trial_13_efficientnetv2s_0.9080, best_f191.5_acc91.5_efficientnetv2s…).
 * Précision = accuracy sur le split test du dataset (2 000 images).
 */
type Stage = {
  model: string;
  role: string;
  specs: string[];
  score: string;
  scoreLabel: string;
  text: string;
  lesson: string;
  tone?: "brand" | "discarded";
};

const stages: Stage[] = [
  {
    model: "MobileNetV3-Large",
    role: "Le point de départ",
    specs: ["5,4 M paramètres", "256 px", "Mac M1"],
    score: "86,7 %",
    scoreLabel: "82,8 % au premier run",
    text: "Léger, conçu pour tourner sur un téléphone, efficace en classification : le candidat naturel pour un détecteur quasi instantané. Transfer learning depuis ImageNet sur un simple Mac M1 — 5 epochs à n'entraîner que la tête, puis dégel complet du backbone à un learning rate de 1e-5. Focal loss et échantillonnage pondéré pour ne favoriser aucune classe, puis quantization int8 : le modèle tient en 4,9 Mo.",
    lesson: "Rapide et minuscule, mais il plafonne sous 87 % malgré le fine-tuning complet.",
  },
  {
    model: "ResNet18",
    role: "Plus de capacité",
    specs: ["11,7 M paramètres", "224 px", "Optuna"],
    score: "87,4 %",
    scoreLabel: "86,3 % au meilleur essai Optuna",
    text: "Deux fois plus de paramètres, une architecture résiduelle éprouvée. Une première recherche Optuna (learning rate, dropout, focal gamma, mixup, scheduler) avec élagage des essais faibles sort le trial #8 à 86,3 %. Fine-tuning ensuite : dégel de layer3 et layer4 à la 3e epoch, backbone 10 fois plus lent que la tête, cosine avec warmup.",
    lesson: "Deux fois plus gros pour +0,7 point : la taille seule ne suffit pas, il faut une architecture qui voit plus fin.",
  },
  {
    model: "EfficientNetV2-S",
    role: "Le changement d'échelle",
    specs: ["21,5 M paramètres", "384 px", "GPU A10G", "20 essais Optuna"],
    score: "90,8 %",
    scoreLabel: "trial #13",
    text: "Un backbone plus récent, entraîné à 384 px — la résolution pour laquelle il a été conçu — et coiffé d'une tête à double pooling : la moyenne des features capte l'allure globale de l'image, le maximum l'artefact local. Vingt essais Optuna (sampler TPE) de 15 epochs, 40 à 55 minutes chacun : learning rate, weight decay, ratio backbone / tête, dropout, label smoothing, nombre de blocs à dégeler et epoch du dégel, scheduler. Hyperband coupe les essais non prometteurs, le SWA stabilise les dernières epochs, et chaque meilleur modèle part automatiquement sur Hugging Face.",
    lesson: "+3,4 points d'un coup : le plus gros saut du projet.",
  },
  {
    model: "Le run à 92 %",
    role: "Le faux record",
    specs: ["écarté"],
    score: "92 %",
    scoreLabel: "sur le jeu de test",
    text: "Un entraînement poussé plus loin affiche 92 %. Mais face à de vraies photos — des selfies smartphone dont le logiciel lisse la peau — il se trompe : il a mémorisé des raccourcis propres au jeu de données (signature du capteur, compression) au lieu d'apprendre les traces de génération. Il est écarté, et on repart du trial #13, une base saine.",
    lesson: "Un point de plus ne vaut rien si le modèle a appris le dataset plutôt que le problème.",
    tone: "discarded",
  },
  {
    model: "EfficientNetV2-S « Universal Real »",
    role: "Le modèle final",
    specs: ["focal loss", "augmentations anti-capteur", "dropout 0,5", "SWA"],
    score: "91,5 %",
    scoreLabel: "et 91,5 % de F1 macro",
    text: "Fine-tuning long depuis le trial #13, avec une seule consigne : généraliser. Focal loss γ = 3 pour se concentrer sur les fakes subtils ; dropout 0,5 et weight decay 1e-3 contre la mémorisation ; BatchNorm dans la tête ; augmentations agressives qui effacent la signature du capteur. Les 4 derniers blocs sont dégelés dès la 2e epoch, le SWA couvre les 7 dernières, l'early stopping surveille le F1 plutôt que la précision.",
    lesson: "Précision et F1 macro identiques : aucune classe n'est sacrifiée pour gonfler le score.",
    tone: "brand",
  },
  {
    model: "Élagué, exporté, servi",
    role: "La mise en production",
    specs: ["pruning L1 20 %", "ONNX FP16", "44 Mo", "sans PyTorch"],
    score: "~0,4 s",
    scoreLabel: "par image, sur CPU",
    text: "Pruning L1 de 20 % des poids des convolutions, export ONNX en FP16, servi par ONNX Runtime dans une API Django sur Railway. Pas de PyTorch en production : le modèle est téléchargé depuis Hugging Face au démarrage, chaque inférence est tracée dans LangSmith, et l'analyse ne coûte aucun appel d'API.",
    lesson: "Le même modèle que celui qui répond dans la démo ci-dessus.",
  },
];

export function Journey() {
  return (
    <Section
      id="parcours"
      index="01"
      eyebrow="Le parcours"
      title={
        <>
          Trois architectures, un faux record, et un modèle qui{" "}
          <span className="accent-italic">généralise</span>.
        </>
      }
      intro="Le challenge : détecter les images de tous les générateurs — Nano Banana Pro comme Midjourney, DALL-E et Stable Diffusion — avec un niveau de confiance fiable, sans confondre une photo retouchée par un smartphone avec une image IA, et répondre quasi instantanément. Les étapes, dans l'ordre où elles ont été franchies."
    >
      <ol className="relative space-y-4 before:absolute before:bottom-8 before:left-[0.6875rem] before:top-8 before:w-px before:bg-hairline-strong">
        {stages.map((s, i) => (
          <li key={s.model} className="relative pl-10">
            <span
              aria-hidden
              className={cn(
                "mono-xs absolute left-0 top-7 grid size-[1.375rem] place-items-center rounded-full border text-[0.65rem]",
                s.tone === "brand"
                  ? "border-brand bg-brand text-on-ink"
                  : s.tone === "discarded"
                    ? "border-hairline-strong bg-paper text-ink-faint"
                    : "border-brand bg-paper text-brand-deep",
              )}
            >
              {i + 1}
            </span>
            <article
              className={cn(
                "grid gap-6 rounded-xl border p-6 sm:grid-cols-[11rem_1fr] sm:p-8",
                s.tone === "brand" ? "border-transparent bg-chart-after/20 shadow-xl" : "border-hairline bg-paper",
                s.tone === "discarded" && "border-dashed",
              )}
            >
              <div>
                <span className="eyebrow">{s.role}</span>
                <div
                  className={cn(
                    "font-display mt-3 text-5xl font-normal tracking-tight tabular-nums",
                    s.tone === "discarded" && "text-ink-faint line-through decoration-2",
                  )}
                >
                  {s.score}
                </div>
                <div className="mono-xs mt-1 text-muted-foreground">{s.scoreLabel}</div>
              </div>
              <div className="min-w-0">
                <h3 className="display-sm">{s.model}</h3>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {s.specs.map((spec) => (
                    <Badge key={spec} variant="mono">
                      {spec}
                    </Badge>
                  ))}
                </div>
                <p className="mt-4 text-sm leading-relaxed text-ink-muted">{s.text}</p>
                <p className="mt-4 border-t border-hairline pt-3 text-sm font-medium">
                  <span className="text-brand-deep">→ </span>
                  {s.lesson}
                </p>
              </div>
            </article>
          </li>
        ))}
      </ol>
    </Section>
  );
}
