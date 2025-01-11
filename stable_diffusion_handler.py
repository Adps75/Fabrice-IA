import requests
import os

# Ajouter la clé API de Replicate dans les variables d'environnement
os.environ["REPLICATE_API_TOKEN"] = "REPLICATE_API_TOKEN"

def generate_image_with_replicate(image_url, general_prompt, elements):
    """
    Génère une image avec Stable Diffusion 3.5 Medium via Replicate.
    Prend en compte un prompt général et une liste d'éléments avec masques et prompts spécifiques.
    """
    try:
        # Étape 1 : Générer l'image avec le prompt général
        print("[DEBUG] Génération de l'image avec le prompt général...")
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers={
                "Authorization": f"Token {os.environ['REPLICATE_API_TOKEN']}",
                "Content-Type": "application/json",
            },
            json={
                "version": "stability-ai/stable-diffusion-3.5-medium",
                "input": {
                    "image": image_url,
                    "prompt": general_prompt,
                },
            },
        )
        response.raise_for_status()
        general_image_url = response.json()["output"]

        # Étape 2 : Appliquer les prompts spécifiques aux masques
        final_image_url = general_image_url
        for element in elements:
            mask_url = element["mask"]
            specific_prompt = element["specific_prompt"]

            print(f"[DEBUG] Application du prompt spécifique : {specific_prompt}")
            response = requests.post(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Token {os.environ['REPLICATE_API_TOKEN']}",
                    "Content-Type": "application/json",
                },
                json={
                    "version": "stability-ai/stable-diffusion-3.5-medium",
                    "input": {
                        "image": final_image_url,
                        "mask": mask_url,
                        "prompt": specific_prompt,
                    },
                },
            )
            response.raise_for_status()
            final_image_url = response.json()["output"]

        return final_image_url

    except Exception as e:
        raise RuntimeError(f"Erreur lors de la génération avec Replicate : {str(e)}")
