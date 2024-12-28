from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import requests
import os
import numpy as np
import cv2
import json  # Import nécessaire
from yolo_handler import predict_objects  # Import de la fonction YOLO

app = Flask(__name__)

@app.route("/")
def home():
    return "Bienvenue sur l'éditeur Python pour Bubble (YOLOv8 + Annotations) !"

# Endpoint pour détecter des objets
@app.route("/detect_objects", methods=["POST"])
def detect_objects():
    """
    Analyse une image pour détecter des objets avec YOLOv8
    et envoie chaque détection séparément à Bubble.
    """
    data = request.json
    image_url = data.get("image_url")
    bubble_save_url = data.get("bubble_save_url")

    if not image_url or not bubble_save_url:
        return jsonify({"success": False, "message": "Paramètres manquants : image_url ou bubble_save_url absent."}), 400

    try:
        # Télécharger l'image
        response = requests.get(image_url)
        response.raise_for_status()
        img_data = BytesIO(response.content)
        pil_image = Image.open(img_data).convert("RGB")
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur téléchargement de l'image : {str(e)}"}), 400

    try:
        # Détecter les objets avec YOLOv8
        detections = predict_objects(pil_image, conf_threshold=0.25)
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur lors de la détection avec YOLOv8 : {str(e)}"}), 500

    # Envoyer chaque détection individuellement à Bubble
    headers = {
        "Authorization": "Bearer bd9d52db77e424541731237a6c6763db",  # Remplacez par la clé API Bubble
        "Content-Type": "application/json"
    }

    for detection in detections:
        payload = {
            "url_image": image_url,
            "class": detection["class"],
            "confidence": detection["confidence"],
            "x1": detection["x1"],
            "y1": detection["y1"],
            "x2": detection["x2"],
            "y2": detection["y2"]
        }

        try:
            bubble_response = requests.post(bubble_save_url, json=payload, headers=headers)
            bubble_response.raise_for_status()
        except Exception as e:
            # Continuer même si une détection échoue
            print(f"Erreur lors de l'envoi de la détection à Bubble : {str(e)}")

    return jsonify({"success": True, "message": "Toutes les détections ont été envoyées à Bubble."})

# Endpoint pour sauvegarder les annotations et générer un masque
@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    """
    Reçoit une image et ses annotations, génère un masque, et envoie les données à Bubble.
    """
    data = request.json
    image_url = data.get("image_url")
    new_annotations = data.get("annotations", [])
    bubble_save_url = data.get("bubble_save_url")

    if not image_url or not bubble_save_url:
        return jsonify({"success": False, "message": "Paramètres manquants : image_url ou bubble_save_url absent."}), 400

    try:
        # Télécharger l'image
        response = requests.get(image_url)
        response.raise_for_status()
        img_data = BytesIO(response.content)
        pil_image = Image.open(img_data).convert("RGB")
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur téléchargement de l'image : {str(e)}"}), 400

    try:
        # Générer le masque
        np_image = np.array(pil_image)
        height, width = np_image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        points = np.array([[int(pt["x"]), int(pt["y"])] for pt in new_annotations])
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 255)

        # Convertir le masque en PNG
        _, mask_png = cv2.imencode('.png', mask)
        mask_bytes = BytesIO(mask_png.tobytes())
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur lors de la création du masque : {str(e)}"}), 500

    # Préparer les données pour Bubble
    payload = {
        "image_url": image_url,
        "annotations": new_annotations
    }
    headers = {
        "Authorization": "Bearer bd9d52db77e424541731237a6c6763db",  # Remplacez par la clé API Bubble
        "Content-Type": "application/json"
    }
    files = {'mask': ('mask.png', mask_bytes, 'image/png')}

    try:
        # Envoyer les résultats à Bubble
        bubble_response = requests.post(bubble_save_url, data=payload, files=files, headers=headers)
        bubble_response.raise_for_status()
        return jsonify({"success": True, "bubble_response": bubble_response.json()})
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur lors de l'envoi à Bubble : {str(e)}"}), 500

if __name__ == "__main__":
    # Port utilisé par Render
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
