# Extension Chrome - FakeFinder

Extension Chrome pour détecter les images générées par IA directement depuis le navigateur.

## Installation

1. Ouvrez Chrome et allez sur `chrome://extensions/`
2. Activez le "Mode développeur" (en haut à droite)
3. Cliquez sur "Charger l'extension non empaquetée"
4. Sélectionnez le dossier `chrome/` de ce projet

## Configuration

1. Cliquez sur l'icône de l'extension dans la barre d'outils
2. Entrez l'URL de votre backend (par défaut: `http://localhost:8000`)
3. Cliquez sur "Enregistrer"

## Utilisation

1. Naviguez sur n'importe quelle page web
2. Cliquez sur une image que vous voulez analyser
3. L'extension envoie l'image au backend pour analyse
4. Le résultat s'affiche avec :
   - Label (FAKE ou REAL)
   - Pourcentage de confiance
   - Heatmap CAM superposée sur l'image

## Développement

### Structure des fichiers

- `manifest.json` - Configuration de l'extension
- `popup.html/js/css` - Interface de configuration
- `content.js/css` - Script injecté dans les pages web
- `background.js` - Service worker pour les tâches en arrière-plan
- `icons/` - Icônes de l'extension (à créer)

### Créer les icônes

Vous devez créer les icônes dans le dossier `icons/` :
- `icon16.png` (16x16)
- `icon48.png` (48x48)
- `icon128.png` (128x128)

Vous pouvez utiliser un générateur d'icônes ou créer vos propres icônes.

## Notes

- L'extension nécessite que le backend Django soit en cours d'exécution
- Par défaut, elle pointe vers `http://localhost:8000`
- L'URL du backend peut être modifiée dans les paramètres de l'extension
