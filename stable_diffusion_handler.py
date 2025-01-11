import replicate

# Ajouter votre clé API Replicate
REPLICATE_API_TOKEN = "r8_Kg6QqJdAWLJk16MIQ5bsuIcyNKi61ri1G28Ao"

def generate_image_with_replicate(prompt):
    """
    Génère une image en utilisant le modèle Stable Diffusion 3.5-medium via Replicate.
    """
    try:
        # Authentification avec la clé API
        replicate.Client(api_token=REPLICATE_API_TOKEN)

        # Modèle utilisé
        model = "stability-ai/stable-diffusion-3.5-medium"

        # Exécution du modèle sur Replicate
        print(f"[DEBUG] Génération d'image avec le prompt : {prompt}")
        output = replicate.run(
            f"{model}:predict",
            input={"prompt": prompt}
        )

        if isinstance(output, list) and len(output) > 0:
            return output[0]  # Retourne l'URL de la première image générée
        else:
            raise RuntimeError("Aucune image générée par le modèle.")

    except Exception as e:
        raise RuntimeError(f"Erreur lors de la génération avec Replicate : {str(e)}")
