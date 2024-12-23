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

let dashOffset = 0;

// Fonction pour récupérer les paramètres de l'URL
function getQueryParam(param) {
    const urlParams = new URLSearchParams(window.location.search);
    return urlParams.get(param);
}

// Charger l'image depuis l'URL
const imageName = getQueryParam("image_url");

if (!imageName) {
    alert("Aucune image spécifiée dans l'URL !");
} else {
    image.src = imageName;
    image.onload = () => {
        setupCanvas();
        resetView();
        redrawCanvas();
    };
}

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
    annotations.forEach(pt => ctx.lineTo(pt.x, pt.y));

    ctx.setLineDash([]);
    ctx.strokeStyle = isLoopClosed() ? "blue" : "red";
    ctx.stroke();

    if (isLoopClosed()) {
        ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
        ctx.fill();
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
    const [dx, dy] = [
        annotations[0].x - annotations[annotations.length - 1].x,
        annotations[0].y - annotations[annotations.length - 1].y,
    ];
    return Math.sqrt(dx ** 2 + dy ** 2) < 10;
}
