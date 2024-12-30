from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import requests
import json
import os
from yolo_handler import predict_objects  # Fonction pour YOLOv8

app = Flask(__name__)
CORS(app)  # Active CORS pour toutes les routes

# Clé API Bubble
API_KEY = "bd9d52db77e424541731237a6c6763db"

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
        # Exemple de détections (simulé)
        detections = [
            {"class": "chair", "confidence": 0.85, "x1": 100, "y1": 200, "x2": 200, "y2": 300},
            {"class": "table", "confidence": 0.95, "x1": 300, "y1": 400, "x2": 400, "y2": 500}
        ]
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur lors de la détection : {str(e)}"}), 500

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    errors = []
    for detection in detections:
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
            "polygon_points": json.dumps(points)
        }

        try:
            bubble_response = requests.post(bubble_save_url, json=payload, headers=headers)
            bubble_response.raise_for_status()
        except Exception as e:
            errors.append(f"Erreur lors de l'envoi à Bubble : {str(e)}")

    if errors:
        return jsonify({"success": False, "message": "Certaines détections ont échoué", "errors": errors}), 207
    return jsonify({"success": True, "message": "Toutes les détections ont été envoyées à Bubble."})

# Endpoint pour sauvegarder les annotations
@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    """
    Reçoit une annotation et l'envoie à Bubble.
    """
    data = request.json
    image_url = data.get("image_url")
    annotations = data.get("annotations", [])
    bubble_save_url = data.get("bubble_save_url")

    if not image_url or not bubble_save_url:
        return jsonify({"success": False, "message": "Paramètres manquants : image_url ou bubble_save_url absent."}), 400

    # Vérification que les annotations sont bien au format attendu
    if not isinstance(annotations, list) or len(annotations) == 0:
        return jsonify({"success": False, "message": "Aucune annotation valide reçue."}), 400

    # Préparation du payload pour Bubble
    points = [{"x": pt["x"], "y": pt["y"]} for pt in annotations]  # Formater les points
    payload = {
        "url_image": image_url,
        "polygon_points": json.dumps(points),  # Convertir en chaîne JSON
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(bubble_save_url, json=payload, headers=headers)
        response.raise_for_status()  # Vérifie les erreurs HTTP
    except requests.exceptions.RequestException as e:
        return jsonify({"success": False, "message": f"Erreur lors de l'envoi à Bubble : {str(e)}"}), 500

    return jsonify({"success": True, "message": "Annotation envoyée à Bubble avec succès."})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
