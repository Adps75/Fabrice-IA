from huggingface_hub import InferenceClient
from PIL import Image
from io import BytesIO

# Ajouter votre clé API Hugging Face
API_TOKEN = "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"

def apply_prompts_with_masks(image_url, general_prompt, elements):
    """
    Applique un prompt général et des prompts spécifiques avec des masques via l'API Hugging Face.
    """
    try:
        # Charger le client d'inférence
        client = InferenceClient(model="stabilityai/stable-diffusion-xl-base-1.0", token=API_TOKEN)

        # Charger l'image originale
        response = requests.get(image_url)
        response.raise_for_status()
        original_image = Image.open(BytesIO(response.content))

        # Appliquer le prompt général
        print("[DEBUG] Appliquer le prompt général...")
        generated_image = client.text_to_image(general_prompt)

        # Boucle pour appliquer les prompts spécifiques avec les masques
        for element in elements:
            mask_url = element["mask"]
            specific_prompt = element["specific_prompt"]

            # Charger le masque
            mask_response = requests.get(mask_url)
            mask_response.raise_for_status()
            mask_image = Image.open(BytesIO(mask_response.content))

            print(f"[DEBUG] Appliquer le prompt spécifique : {specific_prompt}")
            # API d'inférence ne supporte pas directement les masques,
            # vous pouvez ajuster vos requêtes ici si l'API propose des extensions pour cela.

        return generated_image

    except Exception as e:
        raise RuntimeError(f"Erreur lors de la génération avec Stable Diffusion via l'API : {str(e)}")
