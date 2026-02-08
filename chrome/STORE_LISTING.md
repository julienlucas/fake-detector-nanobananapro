# Page Chrome Web Store - FakeFinder

## Informations pour la publication

### Titre
FakeFinder - Détecteur d'Images IA vs Real

### Description courte (132 caractères max)
Détectez les images générées par IA (Midjourney, DALL-E, Stable Diffusion) avec précision. Modèle EfficientNet V2 S, 90% de précision.

### Description complète

**Détectez les images générées par Intelligence Artificielle en un clic**

FakeFinder utilise un modèle de deep learning fine-tuné (EfficientNet V2 S) pour détecter si une image a été générée par une IA ou si elle est réelle.

**🎯 Fonctionnalités principales :**
- ✅ Détection précise des images générées par Midjourney, DALL-E, Stable Diffusion, Nano Banana Pro
- ✅ Analyse en temps réel avec heatmap CAM (Class Activation Mapping)
- ✅ Précision globale : 90% (F1 score 90%, testé sur 2000 images)
- ✅ Précision sur selfies smartphone : 80%
- ✅ Interface simple : glissez-déposez ou sélectionnez une image

**🔬 Technologie :**
- Modèle EfficientNet V2 S entraîné sur 12 000 images variées
- Architecture "Double Pooling" pour détecter les artefacts IA
- Optimisé avec pruning et quantization pour une latence faible
- Inférence via API sécurisée

**📊 Performance :**
- Testé sur 2000 images de validation
- Score F1 macro : 90%
- Précision globale : 90%
- Capable de différencier le pré-traitement logiciel (ex: selfies iPhone) vs génération IA

**💡 Cas d'usage :**
- Vérifier l'authenticité d'images sur les réseaux sociaux
- Détecter les deepfakes et images générées par IA
- Analyser vos propres selfies et comparer avec l'IA
- Identifier les images suspectes avant partage

**🔒 Confidentialité :**
- Les images sont analysées via une API sécurisée
- Aucune donnée n'est stockée
- Respect de la vie privée

Développé avec 💙 par Julien Lucas

### Catégorie
Productivité

### Langue
Français (fr)

### Captures d'écran
Vous devrez créer 1 à 5 captures d'écran :
- Screenshot 1 : Interface principale avec upload d'image
- Screenshot 2 : Résultat d'analyse avec heatmap CAM
- Screenshot 3 : Exemple de détection "FAKE"
- Screenshot 4 : Exemple de détection "REAL"

### Icône de l'extension
- Taille : 128x128 pixels (déjà créée : `icons/apple-icon-144x144.png`)

### Images promotionnelles (optionnel mais recommandé)
- Petite image promotionnelle : 440x280 pixels
- Grande image promotionnelle : 920x680 pixels

### Informations de support
- Email : hello@julienlucas.com
- Site web : https://fakefinder.vercel.app
- Politique de confidentialité : https://fakefinder.vercel.app/privacy (à créer)

### Notes de version
Version 2.0.0 :
- Interface repensée avec upload d'image
- Support du drag & drop
- Affichage des résultats avec heatmap CAM
- Précision améliorée à 90%

## Étapes pour publier sur Chrome Web Store

1. **Créer un compte développeur Chrome Web Store**
   - Aller sur https://chrome.google.com/webstore/devconsole
   - Payer les frais uniques de $5 USD

2. **Préparer le package**
   - Créer un fichier ZIP avec tous les fichiers du dossier `chrome/`
   - Vérifier que le manifest.json est complet
   - Tester l'extension localement

3. **Téléverser l'extension**
   - Cliquer sur "Nouvel élément"
   - Téléverser le fichier ZIP
   - Remplir les informations ci-dessus

4. **Soumettre pour révision**
   - Chrome Web Store examine généralement en 1-3 jours
   - Répondre aux questions si nécessaire

5. **Publication**
   - Une fois approuvée, l'extension sera disponible publiquement
