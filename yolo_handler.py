import os
from ultralytics import YOLO

# Charger le modèle YOLOv8
model_path = "yolov8n.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le fichier {model_path} est introuvable. Placez-le dans le répertoire actuel.")

model = YOLO(model_path)

def predict_objects(image, conf_threshold=0.25):
    """
    Utilise YOLOv8 pour détecter des objets dans une image.
    :param image: PIL.Image.Image, l'image d'entrée.
    :param conf_threshold: float, le seuil de confiance pour les détections.
    :return: list, les détections sous forme de dictionnaire.
    """
    results = model.predict(image, conf=conf_threshold)
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        detections.append({
            "class": class_name,
            "confidence": round(conf, 3),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        })

    return detections
