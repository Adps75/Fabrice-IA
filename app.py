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

# =====================================================================================
# 1) Endpoint pour YOLOv8 : /detect_objects
# =====================================================================================
@app.route("/detect_objects", methods=["POST"])
def detect_objects():
    """
    Analyse une image pour détecter des objets avec YOLOv8.
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
        # Détecter les objets
        detections = predict_objects(pil_image, conf_threshold=0.25)
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur lors de la détection avec YOLOv8 : {str(e)}"}), 500

    # Préparer les données pour Bubble
    payload = {
        "image_url": image_url,
        "detections": json.dumps(detections)
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",  # Clé API Bubble
        "Content-Type": "application/json"
    }

    try:
        # Envoyer les résultats à Bubble
        bubble_response = requests.post(bubble_save_url, json=payload, headers=headers)
        bubble_response.raise_for_status()
        return jsonify({"success": True, "bubble_response": bubble_response.json()})
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur lors de l'envoi à Bubble : {str(e)}"}), 500

# =====================================================================================
# 2) Fonctions utilitaires pour l'annotation et le masque
# =====================================================================================

def calculate_polygon_bounds(points):
    xs = [point['x'] for point in points]
    ys = [point['y'] for point in points]
    return {
        "minX": min(xs),
        "maxX": max(xs),
        "minY": min(ys),
        "maxY": max(ys),
    }

def annotate_image_with_zoom(image_url, points, canvas_size=(800, 600)):
    """
    Génère une version annotée (visuellement) de l'image.
    Renvoie un objet PIL.Image.
    """
    try:
        # Télécharger l'image
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Créer un canvas
        canvas = Image.new("RGB", canvas_size, (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Dessin du polygone (remplissage semi-transparent + contour)
        draw.polygon(points, fill=(255, 0, 0, 128), outline=(255, 0, 0))

        return canvas

    except Exception as e:
        raise ValueError(f"Erreur lors de l'annotation de l'image : {e}")

def generate_mask_from_annotations_no_zoom(image_url, points):
    """
    Crée un masque noir & blanc (zones annotées = blanc) correspondant
    directement aux dimensions originales de l'image.
    """
    try:
        # Télécharger l'image pour récupérer ses dimensions
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Créer une image noire (masque) de la même taille que l'image originale
        mask = Image.new("L", img.size, 0)  # Noir = 0
        draw_mask = ImageDraw.Draw(mask)

        # Dessiner le polygone directement sur le masque
        draw_mask.polygon(
            [(point["x"], point["y"]) for point in points],  # Coordonnées brutes
            fill=255  # Blanc = 255 (zone annotée)
        )

        return mask

    except Exception as e:
        raise ValueError(f"Erreur lors de la génération du masque : {e}")

# =====================================================================================
# 3) Fonction d'upload vers Bubble
# =====================================================================================
def upload_to_bubble_storage(image_buffer, filename="image.png"):
    """
    Upload d'un buffer d'image PNG vers Bubble, renvoie l'URL publique.
    """
    try:
        bubble_upload_url = "https://gardenmasteria.bubbleapps.io/version-test/fileupload"
        files = {
            "file": (filename, image_buffer, "image/png")
        }
        headers = {"Authorization": f"Bearer {API_KEY}"}

        response = requests.post(bubble_upload_url, files=files, headers=headers)
        response.raise_for_status()

        # Nettoyer la réponse : enlever les guillemets et espaces autour
        clean_resp = response.text.strip().strip('"').strip()
        if clean_resp.startswith("//"):
            return f"https:{clean_resp}"
        if clean_resp.startswith("http"):
            return clean_resp

        raise ValueError(f"Réponse inattendue de Bubble : {response.text}")

    except Exception as e:
        raise ValueError(f"Erreur lors de l'upload de l'image sur Bubble : {e}")

# =====================================================================================
# 4) Endpoint principal : /save_annotation
# =====================================================================================
@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    """
    Reçoit les annotations utilisateur, génère :
      - une image annotée
      - un masque noir & blanc
    puis télécharge les deux vers Bubble,
    et renvoie un JSON final.
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
        # Générer l'image annotée
        canvas_size = (800, 600)
        annotated_image = annotate_image_with_zoom(image_url, annotations, canvas_size)

        # Générer le masque (noir & blanc)
        mask_image = generate_mask_from_annotations_no_zoom(image_url, annotations)

        # Uploader l'image annotée
        buffer_annotated = BytesIO()
        annotated_image.save(buffer_annotated, format="PNG")
        buffer_annotated.seek(0)
        annotated_image_url = upload_to_bubble_storage(buffer_annotated, filename="annotated_image.png")

        # Uploader le masque
        buffer_mask = BytesIO()
        mask_image.save(buffer_mask, format="PNG")
        buffer_mask.seek(0)
        mask_image_url = upload_to_bubble_storage(buffer_mask, filename="mask_image.png")

        # Préparer les données pour Bubble ou un autre service
        payload = {
            "url_image": image_url,
            "annotated_image": annotated_image_url,
            "mask_image": mask_image_url,
            "polygon_points": json.dumps(annotations),
            "user": user
        }

        # Envoyer les données au workflow Bubble
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.post(bubble_save_url, json=payload, headers=headers)
        response.raise_for_status()

        return jsonify({
            "success": True,
            "message": "Données et image annotée (et masque) envoyées à Bubble avec succès.",
            "stable_diffusion_input": {
                "base_image_url": image_url,
                "mask_image_url": mask_image_url,
            }
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur : {e}"
        }), 500

# =====================================================================================
# 5) Lancement Flask
# =====================================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
