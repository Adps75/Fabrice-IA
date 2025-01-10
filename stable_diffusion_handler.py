import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from io import BytesIO

# Charger le pipeline de diffusion
def load_pipeline(model_path="stabilityai/stable-diffusion-xl-base"):
    """
    Charge le pipeline de Stable Diffusion pour l'inpainting.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)
    return pipeline

# Fonction pour générer une image
def generate_image(init_image, mask_image, prompt, negative_prompt="", guidance_scale=7.5, num_inference_steps=50):
    """
    Génère une image basée sur une image initiale, un masque et un prompt utilisateur.
    """
    pipeline = load_pipeline()
    
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=init_image,
        mask_image=mask_image,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )

    return result.images[0]  # Retourne l'image générée
