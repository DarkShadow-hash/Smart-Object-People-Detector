import os
from ultralytics import YOLO

# --- Chemins d'accès ---
# Adaptez le chemin d'accès à votre modèle entraîné local (best.pt)
PATH_TO_CUSTOM_MODEL = r'C:\Users\zvelt\OneDrive\Documents\LIFE\AIVANCITY\PGE3 FR\S1\English project\Mon_Projet_Detection\train2_results\weights\best.pt' 

# --- Source Vidéo ---
# 0: Pour utiliser la webcam de votre ordinateur (source=0)
# 'chemin/vers/video.mp4': Pour tester avec un fichier vidéo préenregistré
VIDEO_SOURCE = 0 

# --- Paramètres ---
# Ajuster le seuil de confiance (conf) pour minimiser les fausses détections
CONFIDENCE_THRESHOLD = 0.65 

def test_model_on_video(model_path, source, conf_thresh):
    """
    Charge le modèle entraîné et l'exécute sur une source vidéo (webcam ou fichier).
    
    Args:
        model_path (str): Chemin vers le fichier best.pt.
        source (int/str): 0 pour webcam, ou chemin vers un fichier vidéo.
        conf_thresh (float): Seuil de confiance minimal pour afficher une détection.
    """
    print(f"Chargement du modèle custom : {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Erreur de chargement du modèle : {e}")
        return

    print(f"Lancement de la détection sur source: {source} (Appuyez sur 'q' pour quitter la fenêtre)")
    
    # 1. Lancer la prédiction
    # show=True va ouvrir une fenêtre d'affichage.
    # save=True va sauvegarder l'output (frames) dans runs/detect/
    results = model.predict(
        source=source, 
        show=True, 
        conf=conf_thresh, 
        save=True, 
        name='video_test_output'
    )
    
    print("\nTest terminé. Les résultats sont sauvegardés dans ./runs/detect/video_test_output")


if __name__ == '__main__':
    # Le test doit être exécuté dans l'environnement virtuel actif
    test_model_on_video(PATH_TO_CUSTOM_MODEL, VIDEO_SOURCE, CONFIDENCE_THRESHOLD)
