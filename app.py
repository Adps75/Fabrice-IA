from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image, ImageDraw
import requests
import json
import os
from yolo_handler import predict_objects  # Fonction pour YOLOv8
from stable_diffusion_handler import generate_image_with_replicate # Fonction pour StableDiffusion
import openai

app = Flask(__name__)
CORS(app)  # Active CORS pour toutes les routes

#Bubble API KEY
API_KEY = "bd9d52db77e424541731237a6c6763db"

@app.route("/")
def home():
    return "Bienvenue sur l'éditeur Python pour Bubble (YOLOv8 + Annotations) !"

# =====================================================================================
# Endpoint ChatGPT : /reformulate_prompt (reformule les prompts des utilisateurs)
# =====================================================================================

# Récupère la clé API OpenAI depuis la variable d'environnement sur Render
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/reformulate_prompt", methods=["POST"])
def reformulate_prompt():
    """
    Endpoint pour reformuler un prompt utilisateur en fonction du type de jardin.
    Expects un JSON :
    {
      "user_prompt": "texte saisi par l'utilisateur",
      "garden_type": "Jardin méditerranéen" (exemple)
    }
    """
    try:
        # 1) Extraire les données JSON
        data = request.get_json()
        user_prompt = data.get("user_prompt")
        garden_type = data.get("garden_type")
        system_role = data.get("system_role")

        # 2) Vérifier la présence des champs requis
        if not user_prompt or not garden_type:
            return jsonify({
                "success": False,
                "message": "Paramètres manquants : 'user_prompt' et 'garden_type' sont requis."
            }), 400

        # 3) Appel à l'API OpenAI (ChatGPT)
        # 'role: system' = contexte ; 'role: user' = message utilisateur
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # ou "gpt-4" si vous y avez accès
            messages=[
                {
                    "role": "system",
                    "content": {system_role}
                },
                {
                    "role": "user",
                    "content": f"Type de jardin : {garden_type}.\nPrompt utilisateur : {user_prompt}"
                }
            ],
            max_tokens=200,
            temperature=0.7
        )

        # 4) Extraire la réponse générée
        reformulated_prompt = response["choices"][0]["message"]["content"].strip()

        # 5) Retourner la réponse sous forme JSON
        return jsonify({
            "success": True,
            "reformulated_prompt": reformulated_prompt
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur lors de la reformulation du prompt : {str(e)}"
        }), 500


# =====================================================================================
# 1) Endpoint YOLOv8 : /detect_objects
# =====================================================================================
@app.route("/detect_objects", methods=["POST"])
def detect_objects():
    """
    Analyse une image pour détecter des objets avec YOLOv8.
    Renvoie les prédictions au format JSON et envoie les données à Bubble si nécessaire.
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
        # Détecter les objets via YOLOv8
        detections = predict_objects(pil_image, conf_threshold=0.25)
    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur lors de la détection avec YOLOv8 : {str(e)}"}), 500

    # Préparer les données pour Bubble
    payload = {
        "image_url": image_url,
        "detections": json.dumps(detections)
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
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
# 2) Fonctions utilitaires pour l'annotation et la génération de masque
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

def calculate_zoom_scale(image_width, image_height, bounds, canvas_size):
    polygon_width = bounds["maxX"] - bounds["minX"]
    polygon_height = bounds["maxY"] - bounds["minY"]
    scale_x = canvas_size[0] / polygon_width
    scale_y = canvas_size[1] / polygon_height
    return min(scale_x, scale_y) * 0.9  # Ajouter un padding de 10%

def annotate_image_with_zoom(image_url, points, canvas_size=(800, 600)):
    """
    Génère une version annotée (visuellement) de l'image avec zoom et centrage.
    Renvoie un objet PIL.Image.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        canvas_width, canvas_height = canvas_size

        bounds = calculate_polygon_bounds(points)
        zoom_scale = calculate_zoom_scale(img.width, img.height, bounds, canvas_size)

        center_x = (bounds["minX"] + bounds["maxX"]) / 2
        center_y = (bounds["minY"] + bounds["maxY"]) / 2
        offset_x = (canvas_width / 2) - center_x * zoom_scale
        offset_y = (canvas_height / 2) - center_y * zoom_scale

        img_width = int(img.width * zoom_scale)
        img_height = int(img.height * zoom_scale)
        resized_img = img.resize((img_width, img_height), Image.LANCZOS)

        # Canvas final
        canvas = Image.new("RGB", canvas_size, (255, 255, 255))
        canvas.paste(resized_img, (int(offset_x), int(offset_y)))

        # Overlay des annotations
        overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        # Points mis à l'échelle
        scaled_points = [
            (
                (point["x"] * zoom_scale) + offset_x,
                (point["y"] * zoom_scale) + offset_y
            )
            for point in points
        ]

        # Dessin du polygone
        overlay_draw.polygon(scaled_points, fill=(255, 0, 0, 128))
        overlay_draw.line(scaled_points + [scaled_points[0]], fill=(255, 0, 0, 255), width=5)

        # Fusion overlay
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay)
        return canvas.convert("RGB")

    except Exception as e:
        raise ValueError(f"Erreur lors de l'annotation de l'image : {e}")

def generate_mask(image_url, points):
    """
    Crée un masque noir et blanc où les annotations sont blanches (255) sur fond noir (0).
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Masque noir
        mask = Image.new("L", img.size, 0)
        draw_mask = ImageDraw.Draw(mask)

        # Convertir les points en entiers
        polygon_points = [(int(pt["x"]), int(pt["y"])) for pt in points]
        draw_mask.polygon(polygon_points, fill=255)

        return mask

    except Exception as e:
        raise ValueError(f"Erreur lors de la génération du masque : {e}")


# =====================================================================================
# 3) Upload vers Bubble
# =====================================================================================
def upload_to_bubble_storage(image_buffer, filename="annotated_image.png"):
    try:
        bubble_upload_url = "https://gardenmasteria.bubbleapps.io/version-test/fileupload"
        files = {
            "file": (filename, image_buffer, "image/png")
        }
        headers = {"Authorization": f"Bearer {API_KEY}"}

        print(f"[DEBUG] Envoi du fichier '{filename}' à Bubble...")  # Log debug
        response = requests.post(bubble_upload_url, files=files, headers=headers)
        response.raise_for_status()

        clean_resp = response.text.strip().strip('"').strip()
        print(f"[DEBUG] Réponse brute de Bubble : {repr(clean_resp)}")  # Log debug

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
        # 1) Générer l'image annotée
        canvas_size = (800, 600)
        annotated_image = annotate_image_with_zoom(image_url, annotations, canvas_size)

        # 2) Générer le masque
        mask_image = generate_mask(image_url, annotations)

        # 3) Sauvegarder l'image annotée et le masque dans des buffers
        buffer_annotated = BytesIO()
        annotated_image.save(buffer_annotated, format="PNG")
        buffer_annotated.seek(0)

        buffer_mask = BytesIO()
        mask_image.save(buffer_mask, format="PNG")
        buffer_mask.seek(0)

        # Logs de débogage
        print(f"[DEBUG] Taille du buffer (image annotée): {buffer_annotated.getbuffer().nbytes} octets")
        print(f"[DEBUG] Taille du buffer (masque):       {buffer_mask.getbuffer().nbytes} octets")

        # 4) Uploader sur Bubble
        annotated_image_url = upload_to_bubble_storage(buffer_annotated, filename="annotated_image.png")
        mask_image_url = upload_to_bubble_storage(buffer_mask, filename="mask_image.png")

        # 5) Préparer les données pour Bubble
        payload = {
            "url_image": image_url,
            "annotated_image": annotated_image_url,
            "mask_image": mask_image_url,
            "polygon_points": json.dumps(annotations),
            "user": user
        }

        headers = {"Authorization": f"Bearer {API_KEY}"}

        # 6) Envoyer au workflow Bubble
        response = requests.post(bubble_save_url, json=payload, headers=headers)
        response.raise_for_status()

        return jsonify({
            "success": True,
            "message": "Données, image annotée, et masque envoyés à Bubble avec succès."
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Erreur : {e}"
        }), 500

# =====================================================================================
# 5) Endpoint Stablediffusion /generate_image
# =====================================================================================

@app.route("/generate_image", methods=["POST"])
def generate_image():
    """
    Génère une image avec Stable Diffusion via Replicate.
    """
    try:
        # Récupérer les données entrantes
        data = request.json
        image_url = data.get("image_url")
        general_prompt = data.get("general_prompt")
        elements = data.get("elements", [])  # Liste contenant les masques et prompts spécifiques

        if not image_url or not general_prompt:
            return jsonify({"success": False, "message": "Paramètres manquants : image_url ou prompt général absent."}), 400

        # Appeler la fonction pour générer une image avec Replicate
        generated_image_url = generate_image_with_replicate(image_url, general_prompt, elements)

        # Retourner l'URL de l'image générée
        return jsonify({"success": True, "generated_image_url": generated_image_url})

    except Exception as e:
        return jsonify({"success": False, "message": f"Erreur lors de la génération avec Stable Diffusion : {str(e)}"}), 500

# =====================================================================================
# 6) Lancement de l'application
# =====================================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
