from flask import Flask, request, jsonify
import requests
import os
import base64
import openai

# Initialisation de l'application Flask
app = Flask(__name__)

# Configuration des API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

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
                    "content": f"Tu es un architecte paysagiste collaborant avec une IA. Reformule les prompts pour générer un jardin de style {garden_type} avec des détails réalistes et cohérents."
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

@app.route("/generate_image", methods=["POST"])
def generate_image():
    """
    Génère une image avec Stable Diffusion XL via Replicate.
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

        # Préparer les données pour Stable Diffusion XL
        replicate_payload = {
            "version": "lucataco/sdxl-inpainting",
            "input": {
                "image": image_url,
                "mask": "data:image/png;base64," + mask_base64,
                "prompt": prompt,
                "steps": 50,  # Augmenter pour plus de détails
                "guidance_scale": 7.5,  # Équilibre entre fidélité et créativité
                "seed": 42,  # Pour des résultats reproductibles
                "negative_prompt": "low quality, unrealistic, bad composition, blurry, monochrome",
                "strength": 0.75  # Permet de conserver une bonne structure de l'image d'origine
            }
        }

        headers = {
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        }

        # Appeler l'API Stable Diffusion XL
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=replicate_payload
        )
        response_data = response.json()

        if response.status_code != 200:
            return jsonify({"error": response_data.get("detail", "Erreur API Stable Diffusion.")}), 500

        # Retourner l'URL de l'image générée
        generated_image_url = response_data.get("output", [None])[0]
        if not generated_image_url:
            return jsonify({"error": "Erreur lors de la récupération de l'image générée."}), 500

        return jsonify({
            "success": True,
            "generated_image_url": generated_image_url
        })

    except Exception as e:
        return jsonify({"error": f"Erreur lors de la génération de l'image : {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
