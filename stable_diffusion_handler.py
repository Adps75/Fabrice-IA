from diffusers import StableDiffusionXLInpaintPipeline
import torch
from PIL import Image
import requests
from io import BytesIO

def generate_image(image_url, general_prompt=None, elements=None):
    """
    Fonction qui appelle apply_prompts_with_masks en interne,
    afin de pouvoir être importée dans app.py sans générer d'erreur.
    """
    if elements is None:
        elements = []

    return apply_prompts_with_masks(image_url, general_prompt, elements)

def apply_prompts_with_masks(image_url, general_prompt, elements):
    """
    Applique un prompt général et des prompts spécifiques avec des masques.
    """
    try:
        # Charger le pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)

        # Charger l'image d'origine
        response = requests.get(image_url)
        response.raise_for_status()
        original_image = Image.open(BytesIO(response.content)).convert("RGB")

        # Appliquer le prompt général à l'image entière
        result = pipeline(
            prompt=general_prompt,
            image=original_image
        ).images[0]

        # Appliquer les prompts spécifiques avec les masques
        for element in elements:
            mask_url = element["mask"]
            specific_prompt = element["specific_prompt"]

            # Charger le masque
            mask_response = requests.get(mask_url)
            mask_response.raise_for_status()
            mask_image = Image.open(BytesIO(mask_response.content)).convert("RGB")

            # Appliquer le prompt spécifique au masque
            result = pipeline(
                prompt=specific_prompt,
                image=result,
                mask_image=mask_image
            ).images[0]

        return result

    except Exception as e:
        raise RuntimeError(f"Erreur lors de la génération avec Stable Diffusion : {str(e)}")
