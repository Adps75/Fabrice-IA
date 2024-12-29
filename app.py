from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import requests
import json
import os
from yolo_handler import predict_objects  # Fonction pour YOLOv8

app = Flask(__name__)

API_KEY = "bd9d52db77e424541731237a6c6763db"  # Votre clé API Bubble

@app.route("/")
def home():
    return "Bienvenue sur l'éditeur Python pour Bubble (YOLOv8 + Annotations) !"

# Endpoint pour détecter des objets
@app.route("/detect_objects", methods=["POST"])
def detect_objects():
    """
    Analyse une image pour détecter des objets avec YOLOv8
    et envoie chaque détection avec ses coordonnées en tant que points à Bubble.
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

    # Préparer les en-têtes pour les requêtes vers Bubble
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    errors = []
    for detection in detections:
        # Préparer les points pour le champ `polygon_points`
        points = [
            {"x": detection["x1"], "y": detection["y1"]},
            {"x": detection["x2"], "y": detection["y1"]},
            {"x": detection["x2"], "y": detection["y2"]},
            {"x": detection["x1"], "y": detection["y2"]}
        ]
        payload = {
            "url_image": image_url,
            "class": detection["class"],
            "confidence": detection["confidence"],
            "polygon_points": json.dumps(points)  # Sérialiser la liste de points en JSON
        }

        print(f"Payload envoyé : {json.dumps(payload, indent=2)}")  # Log pour débogage

        try:
            bubble_response = requests.post(bubble_save_url, json=payload, headers=headers)
            bubble_response.raise_for_status()
        except Exception as e:
            errors.append(f"Erreur lors de l'envoi à Bubble : {str(e)}")

    if errors:
        return jsonify({"success": False, "message": "Certaines détections ont échoué", "errors": errors}), 207
    return jsonify({"success": True, "message": "Toutes les détections ont été envoyées à Bubble."})

# Endpoint pour sauvegarder les annotations et générer un masque
@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    """
    Reçoit une image et ses annotations, et envoie chaque annotation avec ses points à Bubble.
    """
    data = request.json
    image_url = data.get("image_url")
    annotations = data.get("annotations", [])  # Liste des annotations sous forme de points
    bubble_save_url = data.get("bubble_save_url")

    if not image_url or not bubble_save_url:
        return jsonify({"success": False, "message": "Paramètres manquants : image_url ou bubble_save_url absent."}), 400

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    errors = []
    for annotation in annotations:
        # Préparer les points pour le champ `polygon_points`
        points = annotation.get("points", [])  # Assurez-vous que chaque annotation contient une liste de points
        if not points:
            errors.append(f"Annotation sans points pour l'image {image_url}")
            continue

        payload = {
            "url_image": image_url,
            "polygon_points": json.dumps(points)  # Sérialiser la liste de points en JSON
        }

        print(f"Payload envoyé : {json.dumps(payload, indent=2)}")  # Log pour débogage

        try:
            bubble_response = requests.post(bubble_save_url, json=payload, headers=headers)
            bubble_response.raise_for_status()
        except Exception as e:
            errors.append(f"Erreur lors de l'envoi à Bubble : {str(e)}")

    if errors:
        return jsonify({"success": False, "message": "Certaines annotations ont échoué", "errors": errors}), 207
    return jsonify({"success": True, "message": "Toutes les annotations ont été envoyées à Bubble."})

if __name__ == "__main__":
    # Port utilisé par Render
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
