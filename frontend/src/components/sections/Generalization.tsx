import { Section } from "@/components/site/primitives";

/*
 * Les augmentations de train_efficient_v2_s.py (« Universal Real »), et pourquoi elles
 * comptent. Chiffres de production : taille du fichier ONNX livré dans models/, latence
 * mesurée en local avec backend/inference_onnx.py.
 */
const augmentations = [
  {
    verb: "Effacer",
    signature: "RandomJPEGCompression(q = 20–80, p = 0.6)",
    text: "La compression des réseaux sociaux, appliquée à 60 % des images : la signature JPEG d'une vraie photo ne suffit plus à la trahir.",
  },
  {
    verb: "Brouiller",
    signature: "GaussianBlur · AdjustSharpness (p = 0.4)",
    text: "Mises au point et optiques variées : flou et netteté aléatoires, pour que le modèle ne s'attache pas au piqué d'un objectif.",
  },
  {
    verb: "Simuler",
    signature: "Autocontrast · ColorJitter · RandomPerspective",
    text: "Le HDR des smartphones et la distorsion d'un selfie pris à bout de bras, face à la focale « portrait » typique des images IA.",
  },
  {
    verb: "Bruiter",
    signature: "GaussianNoise(σ = 0.02, p = 0.5)",
    text: "Un bruit capteur léger ajouté à une image sur deux, réelle ou générée : la présence de bruit ne permet plus de conclure « réel ».",
  },
];

const production = [
  { value: "44 Mo", label: "modèle ONNX FP16", detail: "pruning L1 de 20 % des poids des convolutions" },
  { value: "~0,4 s", label: "par image", detail: "sur CPU (Mac M1), redimension et dessin du cadre compris" },
  { value: "0 €", label: "de coût d'API", detail: "modèle maison servi par ONNX Runtime, sans PyTorch" },
];

const limits = [
  {
    title: "Les selfies smartphone",
    text: "80 % de précision : une peau lissée par le logiciel du téléphone reste le cas le plus proche d'une image IA. Il faut un jeu de selfies plus large, à l'entraînement comme à l'évaluation.",
  },
  {
    title: "Les générateurs de demain",
    text: "Le modèle a vu Nano Banana Pro, Midjourney, DALL-E et Stable Diffusion. Un nouveau générateur peut laisser d'autres traces : le dataset devra suivre.",
  },
  {
    title: "La zone de doute",
    text: "Sous 70 % de confiance, le modèle donne quand même la classe majoritaire. La démo l'affiche comme une présomption plutôt que comme un verdict.",
  },
];

export function Generalization() {
  return (
    <Section
      id="generalisation"
      index="03"
      eyebrow="Le levier déterminant"
      title={
        <>
          Un détecteur ne doit pas reconnaître le capteur : il doit reconnaître{" "}
          <span className="accent-italic">la génération</span>.
        </>
      }
      intro="Toutes les vraies photos d'un dataset sortent d'un appareil, avec son bruit, sa compression, son traitement. Un modèle peut décrocher un bon score en apprenant ces traces-là plutôt que celles de l'IA — c'est ce qu'avait fait le run à 92 % — puis se tromper sur le premier selfie lissé par un iPhone. La parade : effacer la signature du capteur pendant l'entraînement, pour ne laisser au modèle que l'artefact de génération."
    >
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        {augmentations.map((a, i) => (
          <div key={a.signature} className="card-paper border-hairline flex h-full flex-col p-6">
            <div className="flex items-center justify-between">
              <span className="eyebrow">{a.verb}</span>
              <span className="mono-xs text-ink-faint">0{i + 1}</span>
            </div>
            <code className="mono-xs mt-3 w-fit max-w-full break-words rounded-sm bg-brand-surface px-2 py-1 text-brand-deep">
              {a.signature}
            </code>
            <p className="mb-2 mt-3 text-sm leading-relaxed text-ink-muted">{a.text}</p>
          </div>
        ))}
      </div>

      <div className="card-paper border-hairline mt-10 p-6 sm:p-8">
        <p className="display-sm">En production</p>
        <p className="mono-xs mt-1 text-muted-foreground">
          API Django sur Railway · modèle téléchargé depuis Hugging Face au démarrage
        </p>
        <div className="mt-6 grid gap-6 sm:grid-cols-3">
          {production.map((g) => (
            <div key={g.label} className="border-l-2 border-brand pl-4">
              <div className="font-display text-4xl font-normal tracking-tight tabular-nums">{g.value}</div>
              <div className="mt-1 text-sm font-medium">{g.label}</div>
              <div className="mono-xs mt-1 text-muted-foreground">{g.detail}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="mt-14">
        <p className="display-md mt-3">Ce qui limite encore</p>
        <ol className="mt-6 divide-y divide-border border-t border-hairline">
          {limits.map((l, i) => (
            <li key={l.title} className="grid gap-1 py-4 sm:grid-cols-[2rem_16rem_1fr] sm:gap-4">
              <span className="mono-xs pt-1 text-ink-faint">0{i + 1}</span>
              <span className="text-sm font-medium">{l.title}</span>
              <span className="text-sm leading-relaxed text-ink-muted">{l.text}</span>
            </li>
          ))}
        </ol>
        <p className="mt-4 border-t border-sand pt-3 text-xs leading-relaxed text-muted-foreground">
          2 000 images de test : assez pour départager les architectures, trop peu pour garantir le score
          sur n'importe quelle image du web.
        </p>
      </div>
    </Section>
  );
}
