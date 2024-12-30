// Lecture des paramètres d'URL : ?imageUrl=...&bubbleUrl=...
const params = new URLSearchParams(window.location.search);
const imageUrl = params.get("imageUrl") || "";       // URL de l'image
const bubbleSaveUrl = params.get("bubbleUrl") || ""; // Endpoint Bubble pour sauvegarder

if (!imageUrl) {
    console.warn("Aucune imageUrl n'a été fournie. L'image ne pourra pas se charger.");
}

// -- Variables globales --
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");

let annotations = []; // Liste des points (x, y)
let mode = "add"; // "add" pour ajouter des points, "move" pour déplacer l'image
let baseScale = 1.0;
let scale = 1.0;
let offsetX = 0;
let offsetY = 0;
let isDragging = false;
let startX, startY;
let dashOffset = 0; // Décalage pour l'animation des traits pointillés

// Création de l'objet Image
let image = new Image();
if (imageUrl) {
    image.src = imageUrl;
} else {
    image.src = "no_image.png"; // Fallback si pas d'URL
}

// Initialisation du canvas après chargement de l'image
image.onload = () => {
    setupCanvas();
    resetView();
    redrawCanvas();
};

// Redimensionnement de la fenêtre
window.addEventListener('resize', () => {
    setupCanvas();
    resetView();
    redrawCanvas();
});

function setupCanvas() {
    const container = document.querySelector(".canvas-container");
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    const scaleX = canvas.width / image.width;
    const scaleY = canvas.height / image.height;
    baseScale = Math.min(scaleX, scaleY);
}

function resetView() {
    scale = baseScale;
    offsetX = (canvas.width - image.width * scale) / 2;
    offsetY = (canvas.height - image.height * scale) / 2;
}

function redrawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);
    ctx.drawImage(image, 0, 0, image.width, image.height);
    drawAnnotations();
    ctx.restore();
}

function drawAnnotations() {
    if (annotations.length === 0) return;

    ctx.beginPath();
    ctx.lineWidth = 2 / scale;
    ctx.moveTo(annotations[0].x, annotations[0].y);
    for (let i = 1; i < annotations.length; i++) {
        ctx.lineTo(annotations[i].x, annotations[i].y);
    }

    // Si c'est un polygone fermé
    if (isLoopClosed()) {
        ctx.lineTo(annotations[0].x, annotations[0].y);
        ctx.setLineDash([10 / scale, 5 / scale]);
        ctx.lineDashOffset = dashOffset;
        ctx.strokeStyle = "blue";
        ctx.stroke();

        // Remplir le polygone
        ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
        ctx.beginPath();
        ctx.moveTo(annotations[0].x, annotations[0].y);
        for (let i = 1; i < annotations.length; i++) {
            ctx.lineTo(annotations[i].x, annotations[i].y);
        }
        ctx.closePath();
        ctx.fill();
    } else {
        ctx.setLineDash([]);
        ctx.strokeStyle = "red";
        ctx.stroke();
    }

    // Dessiner les points
    annotations.forEach((pt, index) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, index === 0 ? 6 / scale : 4 / scale, 0, 2 * Math.PI);
        ctx.fillStyle = index === 0 ? "blue" : "red";
        ctx.fill();
    });
}

function isLoopClosed() {
    if (annotations.length < 3) return false;
    const dx = annotations[0].x - annotations[annotations.length - 1].x;
    const dy = annotations[0].y - annotations[annotations.length - 1].y;
    return Math.sqrt(dx*dx + dy*dy) < 10;
}

// Animation des pointillés
function animateDashedLine() {
    dashOffset -= 1;
    redrawCanvas();
    requestAnimationFrame(animateDashedLine);
}
animateDashedLine();

// Convertir coordonnées Canvas → Image
function canvasToImageCoords(cx, cy) {
    return {
        x: (cx - offsetX) / scale,
        y: (cy - offsetY) / scale
    };
}

// Clic sur le canvas pour ajouter un point si mode=add
canvas.addEventListener("click", (e) => {
    if (mode === "add") {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const imgCoords = canvasToImageCoords(cx, cy);

        if (imgCoords.x >= 0 && imgCoords.x <= image.width &&
            imgCoords.y >= 0 && imgCoords.y <= image.height) {
            annotations.push({ x: imgCoords.x, y: imgCoords.y });
            redrawCanvas();
        }
    }
});

// Déplacement de l'image (mode=move)
canvas.addEventListener("mousedown", (e) => {
    if (mode === "move") {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        canvas.style.cursor = "grabbing";
    }
});
canvas.addEventListener("mousemove", (e) => {
    if (isDragging) {
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;
        offsetX += dx;
        offsetY += dy;
        startX = e.clientX;
        startY = e.clientY;

        limitOffsets();
        redrawCanvas();
    }
});
canvas.addEventListener("mouseup", () => {
    isDragging = false;
    canvas.style.cursor = (mode === "move") ? "grab" : "crosshair";
});

// Empêcher de sortir du cadre
function limitOffsets() {
    const imgW = image.width * scale;
    const imgH = image.height * scale;

    if (imgW <= canvas.width) {
        offsetX = (canvas.width - imgW) / 2;
    } else {
        if (offsetX > 0) offsetX = 0;
        if (offsetX + imgW < canvas.width) offsetX = canvas.width - imgW;
    }

    if (imgH <= canvas.height) {
        offsetY = (canvas.height - imgH) / 2;
    } else {
        if (offsetY > 0) offsetY = 0;
        if (offsetY + imgH < canvas.height) offsetY = canvas.height - imgH;
    }
}

// Zoom centré
function zoom(factor) {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const before = canvasToImageCoords(centerX, centerY);

    let newScale = scale * factor;
    if (newScale < baseScale) return; // Pas en dessous de l'échelle initiale
    scale = newScale;

    const afterX = before.x * scale + offsetX;
    const afterY = before.y * scale + offsetY;

    offsetX += (centerX - afterX);
    offsetY += (centerY - afterY);

    limitOffsets();
    redrawCanvas();
}

// Gestion des boutons
document.getElementById("addPointsButton").addEventListener("click", () => {
    mode = "add";
    canvas.style.cursor = "crosshair";
});
document.getElementById("moveButton").addEventListener("click", () => {
    mode = "move";
    canvas.style.cursor = "grab";
});
document.getElementById("zoomInButton").addEventListener("click", () => {
    zoom(1.1);
});
document.getElementById("zoomOutButton").addEventListener("click", () => {
    zoom(1 / 1.1);
});
document.getElementById("undoButton").addEventListener("click", () => {
    if (annotations.length > 0) {
        annotations.pop();
        redrawCanvas();
    }
});

// Sauvegarder les annotations
document.getElementById("saveButton").addEventListener("click", () => {
    if (!bubbleSaveUrl) {
        alert("Aucun endpoint Bubble (bubbleUrl) n'est défini !");
        return;
    }
    if (!imageUrl) {
        alert("Aucune imageUrl fournie !");
        return;
    }

    // Envoi des données au backend Flask
    fetch("https://fabrice-ia.onrender.com/save_annotation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            image_url: imageUrl,
            annotations: annotations,
            bubble_save_url: bubbleSaveUrl
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            alert("Annotation sauvegardée avec succès !");
            // Réinitialiser les annotations après sauvegarde
            annotations = [];
            redrawCanvas();
        } else {
            alert("Erreur de sauvegarde : " + data.message);
        }
        console.log("Réponse /save_annotation", data);
    })
    .catch(err => {
        console.error(err);
        alert("Erreur de requête : " + err);
    });
});
