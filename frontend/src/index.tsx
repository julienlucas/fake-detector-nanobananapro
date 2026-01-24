import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Upload, X, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";
import { Ripple } from "@/components/ui/shadcn-io/ripple";
import ContactForm from "@/components/ui/contact-form";

const exampleImages = [
  { value: "/static/04efb32e-e9dc-4389-9b64-ad3af3e3389b_min.webp", label: "Fake Nano Banana Pro" },
  { value: "/static/2c857321-56eb-437d-9abb-f2f97e98628a_min.webp", label: "Fake Nano Banana Pro" },
  { value: "/static/45180048-3fd8-441f-874f-c8a9b56b5b02_min.webp", label: "Fake Nano Banana Pro" },
  { value: "/static/abc85fe7-a3c2-4374-931a-4bee66bd4d9d_min.webp", label: "Fake Nano Banana Pro" },
  { value: "/static/G6GI9FqWMAAuZ_j.jpg", label: "Réel" },
  { value: "/static/GGQ_1723662752223_1723662762427.webp", label: "Réel" },
];

export default function Index() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>("/static/3a4cc2b2-7d83-4b2b-b365-0bf08b9d0c99_min.webp");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{label: string, confidence: number, real_confidence: number, fake_confidence: number, image?: string} | null>(null);
  const [selectedExampleImage, setSelectedExampleImage] = useState<string>("/static/fake-1.png");

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/webp'];
      if (allowedTypes.includes(file.type)) {
        setSelectedFile(file);
        const reader = new FileReader();
        reader.onloadend = () => {
          setPreview(reader.result as string);
        };
        reader.readAsDataURL(file);
        setResult(null);
      }
    }
  };

  const handleRemoveImage = () => {
    setSelectedFile(null);
    setPreview(null);
    setResult(null);
    const input = document.getElementById("file-input") as HTMLInputElement;
    if (input) {
      input.value = "";
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) {
      const allowedTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/webp'];
      if (allowedTypes.includes(file.type)) {
        setSelectedFile(file);
        const reader = new FileReader();
        reader.onloadend = () => {
          setPreview(reader.result as string);
        };
        reader.readAsDataURL(file);
        setResult(null);
      }
    }
  };

  const handleUpload = async () => {
    setLoading(true);
    const body = new FormData();

    try {
      let fileToUpload = selectedFile;

      if (!fileToUpload && preview && preview.startsWith('/static/')) {
        const response = await fetch(preview);
        const blob = await response.blob();
        const fileName = preview.split('/').pop() || 'image.jpg';
        fileToUpload = new File([blob], fileName, { type: blob.type || 'image/jpeg' });
      }

      if (!fileToUpload) {
        console.error('Aucun fichier à uploader');
        setLoading(false);
        return;
      }

      body.append("file", fileToUpload);

      const apiUrl = import.meta.env.VITE_RAILWAY_API_URL || '';
      const response = await fetch(`${apiUrl}/api/inference`, {
        method: "POST",
        body,
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Erreur API:', errorData);
        throw new Error(errorData.error || 'Erreur lors de l\'analyse');
      }

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Erreur:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto max-w-5xl w-full pb-24 px-2">
      <CardHeader className="relative z-20 overflow-hidden">
        <CardTitle
          variant="h1"
          className="relative text-center mx-auto max-w-sm flex items-center justify-center gap-3"
        >
          <img
            src="/static/nanobananapro.webp"
            alt="Nano Banana Pro"
            className="absolute left-0 -mt- rounded-xl w-11 h-11 object-contain"
          />
          <div
            style={{ fontFamily: "'Gabarito', sans-serif" }}
            // bg-gradient-to-br from-green-500 via-slate-900 to-red-700
            className="relative z-20 font-black text-white rounded-xl w-11 h-11 flex items-center justify-center text-3xl"
          >
            F
          </div>
          <span style={{ fontFamily: "'Gabarito', sans-serif" }}>
            Fakefinder
          </span>
        </CardTitle>
        <CardDescription className="text-center text-2xl font-bold text-black">
          Détectez les fakes Nano Banana Pro
        </CardDescription>
        <CardDescription className="text-center text-sm">
          <strong>Modèle fine-tuné</strong> détectant aussi les images de
          diffusion Midjourney, SD et DALL-E
          <br />
          Testé sur 2000 images
          <br />
          Précision: 85%
        </CardDescription>
      </CardHeader>
      <Card className="border-none shadow-none">
        <CardContent className="p-0 mt-8 min-h-[500px] mx-auto space-y-4 text-muted-foreground grid grid-cols-1 md:grid-cols-2 items-stretch justify-center gap-8">
          <div className="flex flex-col h-full">
            <div
              className={cn(
                "relative w-full flex-1 bg-gray-100 border-2 border-dashed rounded-sm text-center transition-colors flex flex-col",
                "border-upload-border hover:border-primary",
                preview ? "p-1" : "p-4 items-center justify-center"
              )}
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              {preview ? (
                <>
                  <img
                    src={preview}
                    alt="Preview"
                    className="w-full h-full object-cover rounded-sm"
                  />
                  <Button
                    variant="secondary"
                    size="icon"
                    className="absolute top-2 right-2 z-10"
                    onClick={handleRemoveImage}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </>
              ) : (
                <>
                  <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-sm text-muted-foreground mb-4">
                    Uploadez une image
                  </p>
                  <input
                    id="file-input"
                    type="file"
                    accept="image/png,image/jpg,image/jpeg,image/webp"
                    className="hidden"
                    onChange={handleFileSelect}
                  />
                  <Button
                    variant="outline"
                    onClick={() =>
                      document.getElementById("file-input")?.click()
                    }
                  >
                    Charger une image
                  </Button>
                </>
              )}
            </div>
            <Button
              onClick={handleUpload}
              className="w-full mt-3"
              size="xl"
              disabled={loading || !preview}
            >
              <Search className="h-4 w-4" />
              {loading ? "Analyse en cours..." : "Analyser l'image"}
            </Button>
          </div>
          <div className="relative -top-4">
            <CardDescription className="mb-4 italic text-sm">
              Exemple de détection (la heatmap indique les zones d'activation)
            </CardDescription>
            <div className="relative rotate-[3deg]">
              <img
                src={
                  result && result.image ? result.image : selectedExampleImage
                }
                alt="Exemple deepfake"
                className="w-full h-auto rounded shadow-lg"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      <Card className="mt-12 border-none mx-auto shadow-none">
        <CardContent className="p-0 border-none">
          <CardTitle variant="h4">Testez avec une de ces images</CardTitle>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
            {exampleImages.map((img) => (
              <div
                key={img.value}
                className={cn(
                  "relative h-20 cursor-pointer rounded-md overflow-hidden transition-all hover:opacity-80",
                  selectedExampleImage === img.value
                    ? "border-primary ring-2 ring-primary"
                    : "border-gray-200 hover:border-gray-300"
                )}
                onClick={() => setPreview(img.value)}
              >
                <img
                  src={img.value}
                  alt={img.label}
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-0 left-0 w-full">
                  <p className="text-white text-sm bg-black/40 p-0.5 px-1">
                    {img.label}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card
        id="contact-form"
        className="mt-12 border-none max-w-2xl mx-auto shadow-none"
      >
        <CardContent className="p-0 border-none">
          <CardTitle
            variant="h2"
            className="bg-gradient-to-br from-black via-black to-black bg-clip-text text-transparent"
          >
            Étude de cas
          </CardTitle>
          <CardTitle variant="h3-card" className="mb-0 mt-4">Le challenge</CardTitle>
          <CardTitle variant="h3" className="font-medium">
            Détecter les fakes avec précision avec un entraînement rapide, peu couteux et
            un modèle léger
          </CardTitle>
          <ul className="list-disc list-inside mb-4 space-y-4">
            <li>
              <strong>
                Pouvoir détecter les images de TOUS les modèles de diffusion.
              </strong> Le modèle ddoit fonctionner sur NanoBanana Pro mais aussi
              Midjourney, Stable Diffusion, DALL-E.
            </li>
            <li>
              <strong>Entraîner un modèle rapidement et à moindre coût.</strong> Pour ça réutiliser les
              connaissances pré-existantes d'un modèle de vision.
            </li>
            <li>
              <strong>Réussir à constiuer un jeu de données Nano Banana Pro par scrapping malgrés peu d'images disponibles.</strong> Puis combiner des datasets
              scrappés Midjourney/DALL-E/SD et Nano Banana Pro pour une
              détection généralisée.
            </li>
            <li>
              <strong>Avoir un modèle faible latence.</strong> Doit pouvoir
              détecter en quelques secondes sur mobile.
            </li>
          </ul>
          <CardTitle variant="h3-card">Résultats et évaluation</CardTitle>
          <ul className="list-inside mb-4 space-y-4">
            <li>
              <strong>🎯 Précision élevée rapidement atteinte</strong> : 8,5/10
              image détectées correctement sur le dataset de test de 2000 images
              Midjourney/DALL-E/SD/Nano Banana Pro
            </li>
            <li>
              <strong>
                ⌛ <span>Entraînement en seulement 3 minutes</span> juste avec
                un Mac Pro M1
              </strong>
              , 1 seule passe sur le jeu de donnée!
            </li>
            <li>
              <strong>
                🧠 Fine-tuning <span>d'un modèle très léger</span>, le
                MobileNetV3 Large par Transfer Learning
              </strong>{" "}
              : tuning de la dernière couche du modèle, le classifieur
              uniquement pour un entraînement ultra rapide et efficace.
            </li>
            <li>
              <strong>
                💰 Zéro coûts d'API lors de l'inférence étant donné que c'est un
                modèle personnel.
              </strong>
            </li>
            <li>
              <strong>
                ⚡ Le modèle a une <span>faible latence</span>.
              </strong>
              <img
                src="/static/langsmith.png"
                alt="LangSmith"
                className="w-full h-auto rounded mt-3 border border-gray-100 rounded-sm"
              />
              <CardDescription className="italic text-center text-xs">
                Montoring dans LangSmith
              </CardDescription>
            </li>
          </ul>
          <p>Et voilà. De bons résultats pour un POC, sans optimisations et avec un modèle très léger pourtant.</p>
          <CardTitle
            variant="h3"
            className="mt-12 max-w-xl mx-auto text-center"
          >
            On discute de votre projet d'automatisation ou d'application?
          </CardTitle>
          <CardDescription className="text-center mb-4">
            Remplissez le formulaire ci-dessous et je vous recontacte dans les
            24-48 heures.
          </CardDescription>
          <ContactForm />
        </CardContent>
      </Card>
    </main>
  );
}