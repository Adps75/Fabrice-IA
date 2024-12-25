from flask import Flask, request, jsonify, send_from_directory, render_template
from io import BytesIO
from PIL import Image
import requests
import os
import numpy as np
import cv2
from yolo_handler import predict_objects  # Import de la fonction YOLO

app = Flask(__name__)

# Dossiers
UPLOAD_FOLDER = "static/images"
MASK_FOLDER = "static/masks"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return "Bienvenue sur l'éditeur Python pour Bubble (YOLOv8 + Annotations) !"

@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    """
    Endpoint pour recevoir une image et ses annotations,
    détecter les objets et créer un masque.
    """
    data = request.json
    image_url = data.get("image_url")
    new_annotations = data.get("annotations", [])
    bubble_save_url = data.get("bubble_save_url")

    if not image_url or not bubble_save_url:
        return jsonify({"success": False, "message": "Paramètres manquants."}), 400

    # Télécharger l'image
    try:
        response = requests.get(image_url)
        response.raise_for_status()
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur téléchargement : {str(e)}"}), 400

    img_data = BytesIO(response.content)
    pil_image = Image.open(img_data).convert("RGB")

    # Détection avec YOLO
    detections = predict_objects(pil_image, conf_threshold=0.25)

    # Générer un masque
    np_image = np.array(pil_image)
    height, width = np_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    points = np.array([[int(pt["x"]), int(pt["y"])] for pt in new_annotations])
    if len(points) >= 3:
        cv2.fillPoly(mask, [points], 255)

    # Sauvegarder le masque
    basename = os.path.basename(image_url).split("?")[0]
    mask_filename = basename.replace(".jpg", ".png").replace(".jpeg", ".png").replace(".png", "_mask.png")
    mask_path = os.path.join(MASK_FOLDER, mask_filename)
    cv2.imwrite(mask_path, mask)

    full_mask_url = request.host_url + "get_mask/" + mask_filename

    # Envoyer à Bubble
    payload = {
        "image_url": image_url,
        "annotations": new_annotations,
        "mask_url": full_mask_url,
        "detections": detections
    }

    try:
        bubble_response = requests.post(bubble_save_url, json=payload)
        bubble_response.raise_for_status()
        return jsonify({"success": True, "bubble_response": bubble_response.json()})
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur envoi Bubble : {str(e)}"}), 500

@app.route("/get_mask/<filename>", methods=["GET"])
def get_mask(filename):
    try:
        return send_from_directory(MASK_FOLDER, filename)
    except FileNotFoundError:
        return jsonify({"success": False, "message": "Masque introuvable."}), 404

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
