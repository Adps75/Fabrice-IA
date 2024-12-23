from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np
import requests

app = Flask(__name__)

# Dossiers
UPLOAD_FOLDER = "static/images"
ICON_FOLDER = "static/icons"
MASK_FOLDER = "static/masks"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# Limite de taille pour les fichiers (16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Dictionnaire pour stocker les annotations
annotations = {}

@app.route("/")
def home():
    app.logger.info("Route principale appelée.")
    return "Bienvenue sur l'éditeur Python pour Bubble !"

# Récupérer une image
@app.route("/get_image/<filename>", methods=["GET"])
def get_image(filename):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(file_path):
            app.logger.error(f"Image non trouvée : {filename}")
            return jsonify({"success": False, "message": "Image non trouvée"}), 404
        return send_from_directory(UPLOAD_FOLDER, filename)
    except Exception as e:
        app.logger.error(f"Erreur lors de l'accès à l'image : {e}")
        return jsonify({"success": False, "message": "Erreur interne"}), 500

# Récupérer une icône (si nécessaire)
@app.route("/get_icon/<filename>", methods=["GET"])
def get_icon(filename):
    try:
        file_path = os.path.join(ICON_FOLDER, filename)
        if not os.path.exists(file_path):
            app.logger.error(f"Icône non trouvée : {filename}")
            return jsonify({"success": False, "message": "Icône non trouvée"}), 404
        return send_from_directory(ICON_FOLDER, filename)
    except Exception as e:
        app.logger.error(f"Erreur lors de l'accès à l'icône : {e}")
        return jsonify({"success": False, "message": "Erreur interne"}), 500

# Éditeur interactif
@app.route("/editor/<image_name>", methods=["GET"])
def editor(image_name):
    try:
        app.logger.info(f"Chargement de l'éditeur pour l'image : {image_name}")
        return render_template("editor.html", image_name=image_name)
    except Exception as e:
        app.logger.error(f"Erreur lors du chargement de l'éditeur : {e}")
        return "Erreur lors du chargement de l'éditeur", 500

# Télécharger une image depuis une URL
def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
        else:
            app.logger.error(f"Erreur lors du téléchargement de l'image : {response.status_code}")
            return False
    except Exception as e:
        app.logger.error(f"Erreur lors du téléchargement de l'image : {e}")
        return False

# Sauvegarder les annotations et générer un masque
@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    try:
        data = request.json
        image_name = data.get("image_name")
        new_annotations = data.get("annotations", [])

        if not image_name or not isinstance(new_annotations, list):
            app.logger.warning("Nom de l'image ou annotations manquants ou incorrects.")
            return jsonify({"success": False, "message": "Nom de l'image ou annotations manquants ou incorrects."}), 400

        # Vérifier chaque annotation
        for annotation in new_annotations:
            if not isinstance(annotation, dict) or "x" not in annotation or "y" not in annotation:
                app.logger.warning("Annotations mal formatées.")
                return jsonify({"success": False, "message": "Annotations mal formatées."}), 400

        if image_name not in annotations:
            annotations[image_name] = []
        annotations[image_name].extend(new_annotations)

        mask_path = generate_mask(image_name)
        if not mask_path:
            app.logger.error("Erreur lors de la génération du masque.")
            return jsonify({"success": False, "message": "Erreur lors de la génération du masque."}), 500

        app.logger.info(f"Annotations sauvegardées et masque généré pour l'image : {image_name}")
        return jsonify({
            "success": True,
            "message": "Annotations enregistrées et masque généré.",
            "annotations": annotations[image_name],
            "mask_path": mask_path
        })
    except Exception as e:
        app.logger.error(f"Erreur lors de la sauvegarde des annotations : {e}")
        return jsonify({"success": False, "message": "Erreur interne"}), 500

# Générer un masque
def generate_mask(image_name):
    try:
        image_annotations = annotations.get(image_name, [])
        if not image_annotations:
            app.logger.warning(f"Aucune annotation trouvée pour l'image : {image_name}")
            return None

        # Vérifier si image_name est une URL
        if image_name.startswith("http://") or image_name.startswith("https://"):
            image_path = os.path.join(UPLOAD_FOLDER, os.path.basename(image_name))
            if not os.path.exists(image_path):
                downloaded = download_image(image_name, image_path)
                if not downloaded:
                    return None
        else:
            image_path = os.path.join(UPLOAD_FOLDER, image_name)

        if not os.path.exists(image_path):
            app.logger.warning(f"Image non trouvée pour générer le masque : {image_name}")
            return None

        image = cv2.imread(image_path)
        if image is None:
            app.logger.error(f"Impossible de lire l'image : {image_name}")
            return None

        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        points = np.array([[int(p["x"]), int(p["y"])] for p in image_annotations])
        cv2.fillPoly(mask, [points], 255)

        mask_filename = f"{os.path.splitext(os.path.basename(image_name))[0]}_mask.png"
        mask_path = os.path.join(MASK_FOLDER, mask_filename)
        cv2.imwrite(mask_path, mask)

        app.logger.info(f"Masque généré : {mask_filename}")
        return f"/static/masks/{mask_filename}"
    except Exception as e:
        app.logger.error(f"Erreur lors de la génération du masque : {e}")
        return None

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Utilisez le port défini par Render ou 5000 par défaut
    app.run(host="0.0.0.0", port=port, debug=True)
