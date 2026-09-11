import { Container, Eyebrow } from "@/components/site/primitives";
import { Detector } from "@/components/detector/Detector";

const tooling = [
  "Reactjs",
  "PyTorch Lightning",
  "Optuna",
  "ONNX Runtime",
  "Django",
  "Hugging Face",
  "LangSmith",
];

const navLinks = [
  { href: "#demo", label: "Démo" },
  { href: "#parcours", label: "Parcours" },
  { href: "#resultats", label: "Résultats" },
  { href: "https://github.com/julienlucas/fake-detector-nanobananapro", label: "Repo GitHub", external: true },
];

export function Hero() {
  return (
    <section id="top" className="border-b border-hairline bg-paper">
      <Container className="py-10 sm:py-16">
        <div className="flex items-start justify-between gap-10">
          <div className="min-w-0">
            <Eyebrow>Étude de cas · Julien Lucas</Eyebrow>
            <h1 className="display-xl mt-4">
              Fake<span className="accent-italic">finder</span>
              <span className="display-md mt-2 block text-ink">
                Différenciateur d'images réels vs images générées par IA — entrâiné sur modèles de diffusion Nano Banana Pro, Midjourney, DALL-E, Stable
                Diffusion — avec 91,5 % de précision sur 2 000 images de test
              </span>
            </h1>
            <p className="copy mt-6">
              Un modèle de vision fine-tuné par transfer learning, parti d'un MobileNetV3 de 5 millions
              de paramètres pour arriver à un EfficientNetV2-S entraîné à ne pas se laisser piéger par
              le capteur d'un smartphone. Entraîné sur un{" "}
              <a
                href="https://huggingface.co/datasets/julienlucas/midjourney-dalle-sd-nanobananapro-dataset"
                target="_blank"
              >
                dataset fait maison de 12 695 images
              </a>{" "}
              , servi en ONNX sans PyTorch et sans coût d'API.
            </p>
          </div>

          <nav aria-label="Sections" className="hidden shrink-0 flex-col items-end gap-3 pt-2 sm:flex">
            {navLinks.map((l) => (
              <a
                key={l.href}
                href={l.href}
                target={l.external ? "_blank" : undefined}
                rel={l.external ? "noreferrer" : undefined}
                className="nav-link mono-xs uppercase"
              >
                {l.label}
                {l.external ? " ↗" : ""}
              </a>
            ))}
          </nav>
        </div>

        <div id="demo" className="scroll-mt-20">
          <Detector />
        </div>

        <div className="mt-8 flex flex-wrap items-center gap-x-4 gap-y-3">
          <Eyebrow>Construit avec</Eyebrow>
          <ul className="flex flex-wrap items-center gap-1.5">
            {tooling.map((t) => (
              <li key={t} className="tag bg-brand-surface">
                {t}
              </li>
            ))}
          </ul>
        </div>
      </Container>
    </section>
  );
}
