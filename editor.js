// On récupère des paramètres depuis l'URL, par exemple ?imageUrl=...&bubbleUrl=...
// Vous pouvez aussi coder ça en dur si vous préférez, ou passer par un attribut data-*.
const params = new URLSearchParams(window.location.search);
const imageUrl = params.get("imageUrl") || "";       // L'URL de l'image sur Bubble
const bubbleSaveUrl = params.get("bubbleUrl") || ""; // L'endpoint Bubble pour sauver

// On vérifie si on a bien une URL
if (!imageUrl) {
    console.warn("Aucune imageUrl n'a été fournie. L'image ne pourra pas se charger.");
}

// Canvas et contexte
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");

// État des annotations
let annotations = [];
let mode = "add"; // "add" ou "move"

// Zoom et déplacement
let baseScale = 1.0;
let scale = 1.0;
let offsetX = 0;
let offsetY = 0;

let isDragging = false;
let startX, startY;
let dashOffset = 0; // Pour l’animation des traits pointillés

// On crée l'objet image
let image = new Image();
if (imageUrl) {
    image.src = imageUrl; // On pointe directement sur l'URL stockée dans Bubble
} else {
    // Si pas d'URL, vous pouvez afficher un placeholder
    image.src = "/static/images/no_image.png";
}

// Une fois l'image chargée, on initialise
image.onload = () => {
    setupCanvas();
    resetView();
    redrawCanvas();
};

// Ajuster la taille du canvas selon la fenêtre
window.addEventListener('resize', () => {
    setupCanvas();
    resetView();
    redrawCanvas();
});

function setupCanvas() {
    const container = document.querySelector(".image-viewer");
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

    // Dessin des annotations (traits + points)
    if (annotations.length > 0) {
        ctx.beginPath();
        ctx.lineWidth = 2 / scale;

        ctx.moveTo(annotations[0].x, annotations[0].y);
        for (let i = 1; i < annotations.length; i++) {
            ctx.lineTo(annotations[i].x, annotations[i].y);
        }

        if (isLoopClosed()) {
            ctx.lineTo(annotations[0].x, annotations[0].y);
            ctx.setLineDash([10 / scale, 5 / scale]);
            ctx.lineDashOffset = dashOffset;
            ctx.strokeStyle = "blue";
            ctx.stroke();

            // Remplir la boucle
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

        // Points
        annotations.forEach((pt, index) => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, index === 0 ? 6 / scale : 4 / scale, 0, 2 * Math.PI);
            ctx.fillStyle = index === 0 ? "blue" : "red";
            ctx.fill();
        });
    }

    ctx.restore();
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

// Ajouter un point sur click
canvas.addEventListener("click", (e) => {
    if (mode === "add") {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const imgCoords = canvasToImageCoords(cx, cy);

        // Vérifier qu'on est dans l'image
        if (imgCoords.x >= 0 && imgCoords.x <= image.width &&
            imgCoords.y >= 0 && imgCoords.y <= image.height) {
            annotations.push({ x: imgCoords.x, y: imgCoords.y });
            redrawCanvas();
        }
    }
});

// Déplacement
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

// Limiter le déplacement (pas sortir du cadre)
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
    if (newScale < baseScale) return; // Pas en dessous du 100%
    scale = newScale;

    const afterX = before.x * scale + offsetX;
    const afterY = before.y * scale + offsetY;

    offsetX += (centerX - afterX);
    offsetY += (centerY - afterY);

    limitOffsets();
    redrawCanvas();
}

// Boutons
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
    zoom(1/1.1);
});

// Annuler le dernier point
document.getElementById("undoButton")?.addEventListener("click", () => {
    if (annotations.length > 0) {
        annotations.pop();
        redrawCanvas();
    }
});

// Sauvegarder
document.getElementById("saveButton").addEventListener("click", () => {
    if (!bubbleSaveUrl) {
        alert("Aucun endpoint Bubble (bubbleUrl) n'est défini !");
        return;
    }
    if (!imageUrl) {
        alert("Aucune imageUrl fournie !");
        return;
    }

    fetch("/save_annotation", {
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
            alert("Sauvegarde OK : " + data.message);
        } else {
            alert("Erreur sauvegarde : " + data.message);
        }
        console.log("Réponse /save_annotation", data);
    })
    .catch(err => {
        console.error(err);
        alert("Erreur de requête : " + err);
    });
});
