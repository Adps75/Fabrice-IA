from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image, ImageDraw

try:
    from PIL import ImageResampling  # Pillow >= 10.0.0
    RESAMPLING = ImageResampling.LANCZOS
except ImportError:
    RESAMPLING = Image.ANTIALIAS  # Versions antérieures

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

# Fonction pour calculer les limites du polygone
def calculate_polygon_bounds(points):
    xs = [point['x'] for point in points]
    ys = [point['y'] for point in points]
    return {
        "minX": min(xs),
        "maxX": max(xs),
        "minY": min(ys),
        "maxY": max(ys),
    }

# Fonction pour calculer le zoom pour inclure le polygone
def calculate_zoom_scale(image_width, image_height, bounds, canvas_size):
    polygon_width = bounds["maxX"] - bounds["minX"]
    polygon_height = bounds["maxY"] - bounds["minY"]
    scale_x = canvas_size[0] / polygon_width
    scale_y = canvas_size[1] / polygon_height
    return min(scale_x, scale_y) * 0.9  # Ajouter un padding de 10%

# Fonction pour annoter l'image avec zoom et centrage
def annotate_image_with_zoom(image_url, annotations, canvas_size=(800, 600)):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Calculer les limites du polygone
        bounds = calculate_polygon_bounds(annotations)

        # Calculer le zoom nécessaire
        zoom_scale = calculate_zoom_scale(img.width, img.height, bounds, canvas_size)

        # Centrer le polygone
        center_x = (bounds["minX"] + bounds["maxX"]) / 2
        center_y = (bounds["minY"] + bounds["maxY"]) / 2
        offset_x = (canvas_size[0] / 2) - center_x * zoom_scale
        offset_y = (canvas_size[1] / 2) - center_y * zoom_scale

        # Redimensionner l'image en appliquant le zoom
        resized_img = img.resize(
            (int(img.width * zoom_scale), int(img.height * zoom_scale)), ImageResampling.LANCZOS
        )

        # Créer un nouveau canvas avec le canvas_size
        canvas = Image.new("RGB", canvas_size, (255, 255, 255))
        canvas.paste(
            resized_img, (int(offset_x), int(offset_y))
        )  # Ajuster selon les offsets

        # Dessiner le polygone
        draw = ImageDraw.Draw(canvas)
        scaled_points = [
            ((point["x"] * zoom_scale) + offset_x, (point["y"] * zoom_scale) + offset_y)
            for point in annotations
        ]

        draw.polygon(scaled_points, outline="red", fill=None, width=3)
        return canvas

    except Exception as e:
        raise ValueError(f"Erreur lors de l'annotation de l'image avec zoom : {str(e)}")

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

def upload_to_bubble_storage(image_buffer, filename="annotated_image.png"):
    """Télécharge l'image vers le stockage Bubble et retourne l'URL publique."""
    try:
        bubble_upload_url = "https://gardenmasteria.bubbleapps.io/version-test/fileupload"  # URL exacte pour le stockage
        files = {
            "file": (filename, image_buffer, "image/png")
        }
        headers = {"Authorization": f"Bearer {API_KEY}"}

        response = requests.post(bubble_upload_url, files=files, headers=headers)
        response.raise_for_status()

        # Vérifiez si la réponse est une chaîne contenant une URL relative
        if response.text.startswith("//"):
            return f"https:{response.text.strip()}"

        # Vérifiez si la réponse est une URL complète
        if response.text.startswith("http"):
            return response.text.strip()

        # Si la réponse est en JSON, essayez de récupérer l'URL
        try:
            response_data = response.json()
            if isinstance(response_data, dict) and "body" in response_data and "url" in response_data["body"]:
                return response_data["body"]["url"]
        except json.JSONDecodeError:
            pass

        # Si aucune des méthodes ne fonctionne, lever une erreur
        raise ValueError(f"Réponse inattendue de Bubble : {response.text}")

    except Exception as e:
        raise ValueError(f"Erreur lors de l'upload de l'image sur Bubble : {str(e)}")

@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    """
    Reçoit les annotations, génère une image annotée,
    télécharge l'image vers Bubble Storage,
    et envoie les données au workflow.
    """
    data = request.json
    image_url = data.get("image_url")
    annotations = data.get("annotations", [])
    user = data.get("user")
    bubble_save_url = data.get("bubble_save_url")

    if not image_url or not bubble_save_url or not user:
        return jsonify({
            "success": False,
            "message": "Paramètres manquants : image_url, bubble_save_url ou user absent."
        }), 400

    if not isinstance(annotations, list) or any("x" not in point or "y" not in point for point in annotations):
        return jsonify({
            "success": False,
            "message": "Annotations non valides."
        }), 400

    try:
        # Générer l'image annotée avec zoom et centrage
        canvas_size = (800, 600)  # Dimensions fixes ou dynamiques selon votre besoin
        annotated_image = annotate_image_with_zoom(image_url, annotations, canvas_size)

        # Sauvegarder l'image annotée dans un buffer
        buffer = BytesIO()
        annotated_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Télécharger sur Bubble Storage
        annotated_image_url = upload_to_bubble_storage(buffer)

        # Préparer les données pour le workflow
        payload = {
            "url_image": image_url,
            "annotated_image": annotated_image_url,  # URL de l'image uploadée
            "polygon_points": json.dumps(annotations),
            "user": user,
        }

        headers = {"Authorization": f"Bearer {API_KEY}"}

        # Envoyer au workflow
        response = requests.post(bubble_save_url, json=payload, headers=headers)
        response.raise_for_status()

        return jsonify({"success": True, "message": "Données et image annotée envoyées à Bubble avec succès."})

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur : {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
