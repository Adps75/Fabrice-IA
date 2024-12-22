// Variables globales
const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");

let image = new Image();
let annotations = [];
let mode = "add"; // "add" ou "move"

let baseScale = 1.0;  
let scale = 1.0;      
let offsetX = 0, offsetY = 0; 

let isDragging = false;
let startX, startY;

let dashOffset = 0; // Pour l'animation des traits pointillés

// Fonction pour récupérer les paramètres de l'URL
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

// Récupérer l'URL de l'image depuis les paramètres
const imageName = getQueryParam("image_url");

if (!imageName) {
    alert("Aucune image spécifiée dans l'URL !");
} else {
    image.src = imageName; // Charger l'image directement depuis l'URL
    image.onload = () => {
        setupCanvas();
        resetView();
        redrawCanvas();
    };
}

// Redimensionner en fonction de la fenêtre
window.addEventListener('resize', () => {
    setupCanvas();
    resetView();
    redrawCanvas();
});

function setupCanvas() {
    const container = document.querySelector(".canvas-container");
    const w = container.clientWidth;
    const h = container.clientHeight;
    canvas.width = w;
    canvas.height = h;

    const scaleX = w / image.width;
    const scaleY = h / image.height;
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

    if (isLoopClosed()) {
        ctx.lineTo(annotations[0].x, annotations[0].y);
        ctx.setLineDash([10 / scale, 5 / scale]);
        ctx.lineDashOffset = dashOffset;
        ctx.strokeStyle = "blue";
        ctx.stroke();

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

    annotations.forEach((pt, i) => {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, i === 0 ? 6 / scale : 4 / scale, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? "blue" : "red";
        ctx.fill();
    });
}

function isLoopClosed() {
    if (annotations.length < 3) return false;
    const dx = annotations[0].x - annotations[annotations.length - 1].x;
    const dy = annotations[0].y - annotations[annotations.length - 1].y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    return dist < 10;
}

function animateDashedLine() {
    dashOffset -= 1;
    redrawCanvas();
    requestAnimationFrame(animateDashedLine);
}
animateDashedLine();

function canvasToImageCoords(cx, cy) {
    return {
        x: (cx - offsetX) / scale,
        y: (cy - offsetY) / scale
    };
}

// Ajouter un point
canvas.addEventListener("click", (e) => {
    if (mode === "add") {
        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        const imgCoords = canvasToImageCoords(cx, cy);

        if (imgCoords.x >= 0 && imgCoords.x <= image.width && imgCoords.y >= 0 && imgCoords.y <= image.height) {
            annotations.push({ x: imgCoords.x, y: imgCoords.y });
            redrawCanvas();
        }
    }
});

canvas.addEventListener("mousedown", (e) => {
    if (mode === "move") {
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        canvas.style.cursor = "grabbing";
    }
});

canvas.addEventListener("mousemove", (e) => {
    if (isDragging && mode === "move") {
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

function limitOffsets() {
    const imgWidth = image.width * scale;
    const imgHeight = image.height * scale;

    if (imgWidth <= canvas.width) {
        offsetX = (canvas.width - imgWidth) / 2;
    } else {
        if (offsetX > 0) offsetX = 0;
        if (offsetX + imgWidth < canvas.width) offsetX = canvas.width - imgWidth;
    }

    if (imgHeight <= canvas.height) {
        offsetY = (canvas.height - imgHeight) / 2;
    } else {
        if (offsetY > 0) offsetY = 0;
        if (offsetY + imgHeight < canvas.height) offsetY = canvas.height - imgHeight;
    }
}

function zoom(factor) {
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const beforeZoom = canvasToImageCoords(centerX, centerY);

    let newScale = scale * factor;
    if (newScale < baseScale) return;
    scale = newScale;

    const afterZoomX = beforeZoom.x * scale + offsetX;
    const afterZoomY = beforeZoom.y * scale + offsetY;

    offsetX += (centerX - afterZoomX);
    offsetY += (centerY - afterZoomY);

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

document.getElementById("undoButton").addEventListener("click", () => {
    if (annotations.length > 0) {
        annotations.pop();
        redrawCanvas();
    }
});

document.getElementById("saveButton").addEventListener("click", () => {
    const maskData = generateMask(); // Placeholder pour la génération du masque

    const annotationData = {
        annotations: annotations,
        mask: maskData
    };

    // Envoyer les données à Bubble
    parent.postMessage(
        {
            type: "save_annotation",
            data: annotationData
        },
        "*"
    );

    alert("Annotation envoyée à Bubble !");
});
