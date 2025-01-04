from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image, ImageDraw
import requests
import json
import os
from yolo_handler import predict_objects  # Fonction pour YOLOv8

app = Flask(__name__)
CORS(app)  # Active CORS pour toutes les routes

API_KEY = "bd9d52db77e424541731237a6c6763db"

@app.route("/")
def home():
    return "Bienvenue sur l'éditeur Python pour Bubble (YOLOv8 + Annotations) !"

# Fonction pour annoter l'image
def annotate_image(image_url, annotations):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Dessiner les polygones
        for annotation in annotations:
            polygon_points = [(pt['x'], pt['y']) for pt in annotation]
            draw.polygon(polygon_points, outline="red", width=5)

        # Sauvegarder l'image en mémoire
        output = BytesIO()
        img.save(output, format="PNG")
        output.seek(0)
        return output

    except Exception as e:
        raise ValueError(f"Erreur lors de l'annotation de l'image : {str(e)}")

# Endpoint pour détecter des objets
@app.route("/detect_objects", methods=["POST"])
def detect_objects():
    """
    Analyse une image pour détecter des objets avec YOLOv8
    et envoie chaque détection avec ses coordonnées (points) à Bubble.
    """
    data = request.json
    image_url = data.get("image_url")
    bubble_save_url = data.get("bubble_save_url")
    user = data.get("user")  # Ajout du paramètre user

    if not image_url or not bubble_save_url or not user:
        return jsonify({"success": False, "message": "Paramètres manquants : image_url, bubble_save_url ou user absent."}), 400

    # Exemple de détection simulée
    detections = [
        {"class": "chair", "confidence": 0.85, "x1": 100, "y1": 200, "x2": 200, "y2": 300},
        {"class": "table", "confidence": 0.95, "x1": 300, "y1": 400, "x2": 400, "y2": 500}
    ]

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
            "polygon_points": json.dumps(points),
            "user": user  # Ajout du user dans le payload
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
    Reçoit les annotations, génère une image annotée,
    et les envoie à Bubble avec l'image.
    """
    data = request.json
    image_url = data.get("image_url")
    annotations = data.get("annotations", [])
    user = data.get("user")  # Ajout du paramètre user
    bubble_save_url = data.get("bubble_save_url") or "https://gardenmasteria.bubbleapps.io/version-test/api/1.1/wf/receive_annotations"

    if not image_url or not bubble_save_url or not user:
        return jsonify({
            "success": False,
            "message": "Paramètres manquants : image_url, bubble_save_url ou user absent."
        }), 400

    if not isinstance(annotations, list) or len(annotations) == 0:
        return jsonify({
            "success": False,
            "message": "Aucune annotation valide reçue."
        }), 400

    try:
        # Télécharger l'image originale
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # Dessiner les annotations sur l'image
        draw = ImageDraw.Draw(image)
        for i, point in enumerate(annotations):
            if i < len(annotations) - 1:
                next_point = annotations[i + 1]
            else:
                next_point = annotations[0]  # Fermer le polygone
            draw.line(
                [(point["x"], point["y"]), (next_point["x"], next_point["y"])],
                fill="red", width=3
            )

        # Sauvegarder l'image annotée dans un buffer
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # Préparer les données pour Bubble
        files = {
            "annotated_image": ("annotated_image.png", buffer, "image/png"),
        }
        payload = {
            "url_image": image_url,
            "polygon_points": json.dumps(annotations),
            "user": user,
        }

        headers = {"Authorization": f"Bearer {API_KEY}"}

        # Envoyer l'image annotée à Bubble
        response = requests.post(bubble_save_url, data=payload, files=files, headers=headers)
        response.raise_for_status()

        return jsonify({"success": True, "message": "Annotation et image envoyées à Bubble avec succès."})

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur lors de la génération ou l'envoi de l'image : {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
