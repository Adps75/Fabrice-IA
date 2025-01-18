from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
from torchvision import models
from torchvision.transforms import functional as F
import requests
import os
import base64
import numpy as np
import openai

# Initialisation de l'application Flask
app = Flask(__name__)

# Configuration des API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Charger le modèle pré-entraîné DeepLabV3
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()  # Mode évaluation pour l'inférence


def process_image_with_deeplab(image_bytes):
    """
    Traite une image avec DeepLabV3 pour générer un masque.
    Retourne l'image d'entrée et le masque binaire correspondant.
    """
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Préparer l'image pour le modèle
    preprocess = T.Compose([
        T.Resize((512, 512)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)

    # Inférence avec le modèle
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
        mask = output.argmax(0).byte().cpu().numpy()

    # Retourne l'image et le masque
    return input_image, mask


@app.route("/reformulate_prompt", methods=["POST"])
def reformulate_prompt():
    """
    Reformule un prompt utilisateur en fonction du type de jardin.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Le corps de la requête est vide."}), 400

        user_prompt = data.get("user_prompt")
        garden_type = data.get("garden_type")

        if not user_prompt or not garden_type:
            return jsonify({"error": "Paramètres manquants : 'user_prompt' et 'garden_type' sont requis."}), 400

        # Appel à l'API OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Tu es un architecte paysagiste collaborant avec une IA. Reformule les prompts pour générer un jardin de style {garden_type}."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        reformulated_prompt = response.choices[0].message.content.strip()
        return jsonify({
            "success": True,
            "reformulated_prompt": reformulated_prompt
        })

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la reformulation du prompt : {str(e)}"}), 500


@app.route("/segment", methods=["POST"])
def segment():
    try:
        # Récupérer l'URL de l'image depuis la requête
        image_url = request.form.get("image_url")

        if not image_url:
            return jsonify({"error": "Aucune URL d'image fournie."}), 400

        # Télécharger l'image depuis l'URL
        response = requests.get(image_url)
        response.raise_for_status()  # Vérifier si la requête a réussi
        image_bytes = response.content

        # Charger l'image avec Pillow
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Préparer l'image pour le modèle
        transformed_image = preprocess_image(image)  # Implémentez cette fonction pour DeepLabV3

        # Effectuer la segmentation avec DeepLabV3
        with torch.no_grad():
            output = model(transformed_image)["out"][0]
            mask = torch.argmax(output, dim=0).byte().cpu().numpy()

        # Convertir le mask en une image PNG
        mask_image = Image.fromarray(mask * 255)  # Multiplier pour avoir une échelle de 0-255
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        buffer.seek(0)

        # Encodage du mask en base64
        encoded_mask = buffer.getvalue().decode("latin1")

        # Retourner le mask au client
        return jsonify({
            "success": True,
            "mask": encoded_mask
        })

    except requests.exceptions.RequestException as req_err:
        return jsonify({"error": f"Erreur lors du téléchargement de l'image : {str(req_err)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Erreur lors du traitement de l'image : {str(e)}"}), 500


@app.route("/generate_image", methods=["POST"])
def generate_image():
    """
    Génère une image avec Stable Diffusion via Replicate.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Le corps de la requête est vide."}), 400

        image_url = data.get("image_url")
        mask_base64 = data.get("mask_base64")
        prompt = data.get("prompt")

        if not image_url or not mask_base64 or not prompt:
            return jsonify({"error": "Paramètres manquants : 'image_url', 'mask_base64', ou 'prompt'."}), 400

        # Convertir le masque base64 en bytes
        mask_bytes = base64.b64decode(mask_base64)

        # Préparer les données pour Stable Diffusion
        replicate_payload = {
            "version": "stability-ai/stable-diffusion-inpainting",
            "input": {
                "image": image_url,
                "mask": mask_bytes.decode('latin1'),  # Convertir en format compatible
                "prompt": prompt
            }
        }
        headers = {
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        }

        # Appeler l'API Stable Diffusion
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=replicate_payload
        )
        response_data = response.json()

        if response.status_code != 200:
            return jsonify({"error": response_data.get("detail", "Erreur API Stable Diffusion.")}), 500

        # Retourner l'URL de l'image générée
        generated_image_url = response_data["output"]
        return jsonify({
            "success": True,
            "generated_image_url": generated_image_url
        })

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image : {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
