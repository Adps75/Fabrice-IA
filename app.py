from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

# YOLOv8
from ultralytics import YOLO

app = Flask(__name__)

# Dossiers
UPLOAD_FOLDER = "static/images"
MASK_FOLDER = "static/masks"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# Charger un modèle YOLOv8 pré-entraîné
model = YOLO("yolov8n.pt")  # Remplacez par un modèle personnalisé si nécessaire

# Stockage temporaire pour les annotations
annotations_store = {}

@app.route("/")
def home():
    return "Bienvenue sur l'éditeur Python pour Bubble (YOLOv8 + Annotations) !"

@app.route("/editor", methods=["GET"])
def editor():
    return render_template("editor.html")

@app.route("/get_image/<filename>", methods=["GET"])
def get_image(filename):
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except FileNotFoundError:
        return jsonify({"success": False, "message": "Image non trouvée"}), 404

@app.route("/get_mask/<filename>", methods=["GET"])
def get_mask(filename):
    try:
        return send_from_directory(MASK_FOLDER, filename)
    except FileNotFoundError:
        return jsonify({"success": False, "message": "Masque non trouvé"}), 404

@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    """
    Reçoit un JSON avec :
    - image_url : URL de l'image (provenant de Bubble ou autre)
    - annotations : liste de points (x, y)
    - bubble_save_url : endpoint Bubble pour sauvegarder les résultats

    Étapes :
      1) Télécharger l'image depuis l'URL
      2) Analyser l'image avec YOLOv8 (détection des objets)
      3) Générer un masque à partir des annotations
      4) Envoyer les résultats à Bubble
    """
    data = request.json
    image_url = data.get("image_url")
    new_annotations = data.get("annotations", [])
    bubble_save_url = data.get("bubble_save_url")

    if not image_url or not bubble_save_url:
        return jsonify({"success": False,
                        "message": "Paramètres manquants : image_url et bubble_save_url sont requis."}), 400

    # Sauvegarde locale des annotations
    annotations_store[image_url] = new_annotations

    # 1) Télécharger l'image
    try:
        r = requests.get(image_url)
        r.raise_for_status()
    except Exception as e:
        return jsonify({"success": False, "message": f"Impossible de télécharger l'image : {str(e)}"}), 400

    # Conversion en PIL
    img_data = BytesIO(r.content)
    pil_image = Image.open(img_data).convert("RGB")

    # 2) YOLOv8 : détection
    results = model.predict(pil_image, conf=0.25)  # Confiance minimum pour les détections
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        detections.append({
            "class": class_name,
            "confidence": round(conf, 3),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        })

    # 3) Générer le masque à partir des annotations
    np_image = np.array(pil_image)
    height, width = np_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    points = np.array([[int(pt["x"]), int(pt["y"])] for pt in new_annotations])
    if len(points) >= 3:
        cv2.fillPoly(mask, [points], 255)

    # Stocker le masque localement
    basename = os.path.basename(image_url).split("?")[0]
    mask_filename = basename.replace(".jpg", ".png").replace(".jpeg", ".png").replace(".png", "_mask.png")
    mask_path = os.path.join(MASK_FOLDER, mask_filename)
    cv2.imwrite(mask_path, mask)

    full_mask_url = request.host_url + "get_mask/" + mask_filename

    # 4) Envoyer le résultat complet à Bubble
    payload_bubble = {
        "image_url": image_url,
        "annotations": new_annotations,
        "mask_url": full_mask_url,
        "detections": detections
    }

    try:
        bubble_response = requests.post(bubble_save_url, json=payload_bubble)
        bubble_response.raise_for_status()
        bubble_json = bubble_response.json()
    except Exception as ex:
        return jsonify({
            "success": False,
            "message": f"Echec envoi à Bubble : {str(ex)}"
        }), 500

    return jsonify({
        "success": True,
        "message": "Annotations & YOLOv8 envoyées à Bubble avec succès.",
        "mask_path": full_mask_url,
        "bubble_response": bubble_json,
        "detections": detections
    })

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
