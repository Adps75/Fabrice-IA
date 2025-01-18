from flask import Flask, request, jsonify
from PIL import Image
import io
import torch
import torchvision
from torchvision.transforms import functional as F
import openai
import requests
import os

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle pré-entraîné Mask R-CNN
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Mode évaluation pour l'inférence

# Configuration des clés API
openai.api_key = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def process_image(image_bytes):
    """
    Prépare l'image pour le modèle Mask R-CNN.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = F.to_tensor(image)  # Convertir l'image en tenseur (C, H, W)
    return image, image_tensor

@app.route("/process", methods=["POST"])
def process():
    """
    Endpoint pour traiter une image avec Mask R-CNN.
    """
    try:
        # Vérifier si le fichier est fourni
        if 'image' not in request.files:
            return jsonify({"error": "Aucune image fournie."}), 400

        # Lire l'image et la convertir en tenseur
        image_file = request.files['image']
        image_bytes = image_file.read()
        original_image, image_tensor = process_image(image_bytes)

        # Ajouter une dimension batch et effectuer l'inférence
        with torch.no_grad():
            predictions = model([image_tensor])

        # Traiter les résultats
        result = []
        for i, box in enumerate(predictions[0]['boxes']):
            score = predictions[0]['scores'][i].item()
            if score >= 0.5:  # Seulement les détections avec une confiance >= 50%
                box = box.tolist()  # Convertir les coordonnées en liste
                mask = predictions[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
                result.append({
                    "box": box,
                    "score": score,
                    "mask": mask.tolist(),
                })

        return jsonify({"success": True, "predictions": result})

    except Exception as e:
        return jsonify({"error": f"Erreur lors du traitement de l'image : {str(e)}"}), 500

@app.route("/generate_image", methods=["POST"])
def generate_image():
    """
    Endpoint pour générer une image en utilisant Stable Diffusion.
    """
    try:
        # Extraire les données JSON
        data = request.get_json()
        image_url = data.get("image_url")
        garden_style = data.get("garden_style")
        user_prompt = data.get("user_prompt")

        if not image_url or not garden_style or not user_prompt:
            return jsonify({"error": "Paramètres manquants : 'image_url', 'garden_style', et 'user_prompt' sont requis."}), 400

        # Reformuler le prompt utilisateur avec OpenAI
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Tu es un architecte paysagiste collaborant avec une IA de génération d'images. Ta mission est de traduire les idées des utilisateurs en prompts précis pour générer un jardin de style {garden_style}."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )
        reformulated_prompt = openai_response.choices[0].message.content.strip()

        # Préparer les données pour l'API de Stable Diffusion
        replicate_payload = {
            "version": "stability-ai/stable-diffusion-inpainting",
            "input": {
                "image": image_url,
                "mask": mask_url,
                "prompt": reformulated_prompt
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
            return jsonify({"error": response_data.get("error", "Erreur API Stable Diffusion")}), 500

        # Retourner l'image générée
        generated_image_url = response_data["output"]
        return jsonify({"success": True, "generated_image_url": generated_image_url})

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image : {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
