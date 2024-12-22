from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np

app = Flask(__name__)

# Dossiers
UPLOAD_FOLDER = "static/images"
ICON_FOLDER = "static/icons"
MASK_FOLDER = "static/masks"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# Dictionnaire pour stocker les annotations
annotations = {}

@app.route("/")
def home():
    return "Bienvenue sur l'éditeur Python pour Bubble !"

# Récupérer une image
@app.route("/get_image/<filename>", methods=["GET"])
def get_image(filename):
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except FileNotFoundError:
        return jsonify({"success": False, "message": "Image non trouvée"}), 404

# Récupérer une icône (si nécessaire)
@app.route("/get_icon/<filename>", methods=["GET"])
def get_icon(filename):
    try:
        return send_from_directory(ICON_FOLDER, filename)
    except FileNotFoundError:
        return jsonify({"success": False, "message": "Icône non trouvée"}), 404

# Éditeur interactif
@app.route("/editor/<image_name>", methods=["GET"])
def editor(image_name):
    # Rendre le template avec le nom de l'image
    return render_template("editor.html", image_name=image_name)

# Sauvegarder les annotations et générer un masque
@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    data = request.json
    image_name = data.get("image_name")
    new_annotations = data.get("annotations", [])

    if not image_name or not new_annotations:
        return jsonify({"success": False, "message": "Nom de l'image ou annotations manquants."}), 400

    if image_name not in annotations:
        annotations[image_name] = []
    annotations[image_name].extend(new_annotations)

    mask_path = generate_mask(image_name)
    if not mask_path:
        return jsonify({"success": False, "message": "Erreur lors de la génération du masque."}), 500

    return jsonify({
        "success": True,
        "message": "Annotations enregistrées et masque généré.",
        "annotations": annotations[image_name],
        "mask_path": mask_path
    })

def generate_mask(image_name):
    image_annotations = annotations.get(image_name, [])
    if not image_annotations:
        return None

    image_path = os.path.join(UPLOAD_FOLDER, image_name)
    if not os.path.exists(image_path):
        return None

    image = cv2.imread(image_path)
    if image is None:
        return None

    height, width, _ = image.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    points = np.array([[int(p["x"]), int(p["y"])] for p in image_annotations])
    cv2.fillPoly(mask, [points], 255)

    mask_filename = f"{os.path.splitext(image_name)[0]}_mask.png"
    mask_path = os.path.join(MASK_FOLDER, mask_filename)
    cv2.imwrite(mask_path, mask)

    return f"/static/masks/{mask_filename}"

if __name__ == "__main__":
    app.run(debug=True)
