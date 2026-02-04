import os
from huggingface_hub import HfApi, login
from dotenv import load_dotenv

load_dotenv()

def upload_script_to_hf(repo_id="julienlucas/fakefinder", file_path=None):
    """Upload le script optimize_hyperparameters_hf.py vers Hugging Face"""
    
    if file_path is None:
        file_path = os.path.join(os.path.dirname(__file__), "optimize_hyperparameters_hf.py")
    
    if not os.path.exists(file_path):
        print(f"❌ Fichier non trouvé: {file_path}")
        return
    
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_ACCESS_TOKEN")
    
    if not token:
        print("❌ Token Hugging Face non trouvé dans les variables d'environnement")
        print("   Définissez HF_TOKEN, HUGGING_FACE_HUB_TOKEN ou HF_ACCESS_TOKEN")
        return
    
    try:
        print(f"🔐 Connexion à Hugging Face...")
        login(token=token.strip())
        
        api = HfApi(token=token.strip())
        
        print(f"📦 Création du repo {repo_id} si nécessaire...")
        try:
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
        except Exception as e:
            print(f"   (Repo existe déjà ou erreur: {e})")
        
        print(f"📤 Upload de {file_path} vers {repo_id}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo="optimize_hyperparameters_hf.py",
            repo_id=repo_id,
            repo_type="model",
            token=token.strip()
        )
        
        print(f"✅ Fichier uploadé avec succès: {repo_id}/optimize_hyperparameters_hf.py")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'upload: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload optimize_hyperparameters_hf.py vers Hugging Face")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="julienlucas/fakefinder",
        help="ID du repo Hugging Face"
    )
    parser.add_argument(
        "--file-path",
        type=str,
        default=None,
        help="Chemin vers le fichier à uploader (défaut: optimize_hyperparameters_hf.py dans le même dossier)"
    )
    
    args = parser.parse_args()
    
    upload_script_to_hf(repo_id=args.repo_id, file_path=args.file_path)
