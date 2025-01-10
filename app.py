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
# Fonctions utilitaires
# =====================================================================================

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
def annotate_image_with_zoom(image_url, points, canvas_size=(800, 600)):
    try:
        # Télécharger l'image
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Dimensions du canvas
        canvas_width, canvas_height = canvas_size

        # Calcul des limites du polygone
        bounds = calculate_polygon_bounds(points)

        # Calculer le zoom
        zoom_scale = calculate_zoom_scale(img.width, img.height, bounds, canvas_size)

        # Centrer le polygone
        center_x = (bounds["minX"] + bounds["maxX"]) / 2
        center_y = (bounds["minY"] + bounds["maxY"]) / 2

        offset_x = (canvas_width / 2) - center_x * zoom_scale
        offset_y = (canvas_height / 2) - center_y * zoom_scale

        # Redimensionner l'image
        img_width = int(img.width * zoom_scale)
        img_height = int(img.height * zoom_scale)
        resized_img = img.resize((img_width, img_height), Image.LANCZOS)

        # Créer le canvas
        canvas = Image.new("RGB", canvas_size, (255, 255, 255))
        canvas.paste(resized_img, (int(offset_x), int(offset_y)))

        # Créer une couche temporaire pour la transparence
        overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Points scalés pour le zoom
        scaled_points = [
            (
                (point["x"] * zoom_scale) + offset_x,
                (point["y"] * zoom_scale) + offset_y
            )
            for point in points
        ]

        # Dessiner le remplissage semi-transparent sur l'overlay
        overlay_draw.polygon(scaled_points, fill=(255, 0, 0, 128))

        # Dessiner le contour sur l'overlay
        overlay_draw.line(scaled_points + [scaled_points[0]], fill=(255, 0, 0, 255), width=5)

        # Combiner l'overlay avec le canvas principal
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay)

        return canvas.convert("RGB")  # Retourner au mode RGB si nécessaire

    except Exception as e:
        raise ValueError(f"Erreur lors de l'annotation de l'image : {e}")

# Fonction pour générer un masque noir et blanc
def generate_mask(image_url, points):
    """
    Crée un masque noir et blanc où les annotations sont blanches (255) sur fond noir (0).
    """
    try:
        # Télécharger l'image pour récupérer ses dimensions
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Créer une image noire de la même taille que l'image originale
        mask = Image.new("L", img.size, 0)  # Noir (0)
        draw_mask = ImageDraw.Draw(mask)

        # Dessiner le polygone directement sur le masque
        polygon_points = [(point["x"], point["y"]) for point in points]
        draw_mask.polygon(polygon_points, fill=255)  # Blanc (255)

        return mask

    except Exception as e:
        raise ValueError(f"Erreur lors de la génération du masque : {e}")

# =====================================================================================
# Fonction pour uploader une image vers Bubble Storage
# =====================================================================================
def upload_to_bubble_storage(image_buffer, filename="annotated_image.png"):
    try:
        bubble_upload_url = "https://gardenmasteria.bubbleapps.io/version-test/fileupload"
        files = {
            "file": (filename, image_buffer, "image/png")
        }
        headers = {"Authorization": f"Bearer {API_KEY}"}

        response = requests.post(bubble_upload_url, files=files, headers=headers)
        response.raise_for_status()

        clean_resp = response.text.strip().strip('"').strip()
        if clean_resp.startswith("//"):
            return f"https:{clean_resp}"
        if clean_resp.startswith("http"):
            return clean_resp

        raise ValueError(f"Réponse inattendue de Bubble : {response.text}")

    except Exception as e:
        raise ValueError(f"Erreur lors de l'upload de l'image sur Bubble : {e}")

# =====================================================================================
# Endpoint principal
# =====================================================================================
@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    """
    Reçoit les annotations, génère une image annotée et un masque,
    télécharge les deux sur Bubble Storage, et envoie les données au workflow.
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

        # Générer le masque
        mask_image = generate_mask(image_url, annotations)

        # Sauvegarder l'image annotée et le masque dans des buffers
        buffer_annotated = BytesIO()
        annotated_image.save(buffer_annotated, format="PNG")
        buffer_annotated.seek(0)

        buffer_mask = BytesIO()
        mask_image.save(buffer_mask, format="PNG")
        buffer_mask.seek(0)

        # Télécharger sur Bubble Storage
        annotated_image_url = upload_to_bubble_storage(buffer_annotated, filename="annotated_image.png")
        mask_image_url = upload_to_bubble_storage(buffer_mask, filename="mask_image.png")

        # Préparer les données pour le workflow
        payload = {
            "url_image": image_url,
            "annotated_image": annotated_image_url,
            "mask_image": mask_image_url,
            "polygon_points": json.dumps(annotations),
            "user": user,
        }

        headers = {"Authorization": f"Bearer {API_KEY}"}

        # Envoyer au workflow
        response = requests.post(bubble_save_url, json=payload, headers=headers)
        response.raise_for_status()

        return jsonify({"success": True, "message": "Données, image annotée, et masque envoyés à Bubble avec succès."})

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur : {e}"
        }), 500

# =====================================================================================
# Lancement de l'application
# =====================================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
