from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import requests
from io import BytesIO
import os

app = FastAPI()

# Configurez votre clé API OpenAI et Stable Diffusion
openai.api_key = "OPENAI_API_KEY"
REPLICATE_API_TOKEN = "REPLICATE_API_TOKEN"

class GenerateImageRequest(BaseModel):
    image_url: str
    garden_style: str
    user_prompt: str

@app.post("/generate_image")
async def generate_image(request: GenerateImageRequest):
    try:
        # Étape 1 : Reformuler le prompt utilisateur avec OpenAI
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Tu es un architecte paysagiste collaborant avec une IA de génération d'images. Ta mission est de traduire les idées des utilisateurs en prompts précis pour générer un jardin de style {request.garden_style}."
                },
                {
                    "role": "user",
                    "content": request.user_prompt
                }
            ]
        )
        reformulated_prompt = openai_response.choices[0].message.content.strip()

        # Étape 2 : Préparer les données pour Stable Diffusion
        replicate_payload = {
            "version": "stability-ai/stable-diffusion-inpainting",
            "input": {
                "image": request.image_url,
                "mask": None,  # Pas de masque pour l'instant
                "prompt": reformulated_prompt
            }
        }
        headers = {
            "Authorization": f"Token {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        }

        # Étape 3 : Appeler l'API Stable Diffusion
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=replicate_payload
        )
        response_data = response.json()

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=response_data.get("error", "Erreur API Stable Diffusion"))

        # Étape 4 : Retourner l'image générée
        generated_image_url = response_data["output"]
        return {"success": True, "generated_image_url": generated_image_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de l'image : {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
