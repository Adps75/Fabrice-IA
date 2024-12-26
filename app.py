from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import requests
import os
import numpy as np
import cv2
from yolo_handler import predict_objects  # Import de la fonction YOLO

app = Flask(__name__)

@app.route("/")
def home():
    return "Bienvenue sur l'éditeur Python pour Bubble (YOLOv8 + Annotations) !"

@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    """
    Endpoint pour recevoir une image et ses annotations,
    détecter les objets avec YOLOv8 et créer un masque basé sur les annotations.
    """
    # Vérification des données entrantes
    data = request.json
    image_url = data.get("image_url")
    new_annotations = data.get("annotations", [])
    bubble_save_url = data.get("bubble_save_url")

    if not image_url or not bubble_save_url:
        return jsonify({"success": False, "message": "Paramètres manquants : image_url ou bubble_save_url absent."}), 400

    # Télécharger l'image depuis Bubble
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img_data = BytesIO(response.content)
        pil_image = Image.open(img_data).convert("RGB")
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur téléchargement de l'image : {str(e)}"}), 400

    # Détection des objets avec YOLOv8
    try:
        detections = predict_objects(pil_image, conf_threshold=0.25)
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur lors de la détection avec YOLOv8 : {str(e)}"}), 500

    # Générer le masque basé sur les annotations
    try:
        np_image = np.array(pil_image)
        height, width = np_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        points = np.array([[int(pt["x"]), int(pt["y"])] for pt in new_annotations])
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)

        # Convertir le masque en format PNG
        _, mask_png = cv2.imencode('.png', mask)
        mask_bytes = BytesIO(mask_png.tobytes())
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur lors de la création du masque : {str(e)}"}), 500

    # Envoyer les résultats à Bubble
    payload = {
        "image_url": image_url,
        "annotations": new_annotations,
        "detections": detections
    }
    files = {'mask': ('mask.png', mask_bytes, 'image/png')}
    headers = {
        "Authorization": "Bearer bd9d52db77e424541731237a6c6763db",  # Remplacez YOUR_API_KEY par la clé API de Bubble
        "Content-Type": "application/json"
    }

    try:
        bubble_response = requests.post(bubble_save_url, json=payload, files=files, headers=headers)
        bubble_response.raise_for_status()
        return jsonify({"success": True, "bubble_response": bubble_response.json()})
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur lors de l'envoi à Bubble : {str(e)}"}), 500

if __name__ == "__main__":
    # Récupérer le port défini par Render
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
