const morphCanvas = document.createElement('canvas');
const morphCtx = morphCanvas.getContext('2d');
const processedImage = document.getElementById("processedImage");
const video = document.getElementById('webcam');
const promptInput = document.getElementById('prompt');
const latencyDisplay = document.getElementById('latency');
const errorBox = document.getElementById('error');
const nr_gpus = 4;

let oldImageData = null;
let delay = 1000;
let avg_latency = [];

async function startWebcam() {
	try {
		const stream = await navigator.mediaDevices.getUserMedia({ video: true });
		video.srcObject = stream;
	} catch (err) {
		showError("Kein Zugriff auf Kamera: " + err.message);
	}
}

async function getFrameBlob(video) {
	const canvas = document.createElement('canvas');
	canvas.width = video.videoWidth;
	canvas.height = video.videoHeight;
	const ctx = canvas.getContext('2d');
	ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
	return new Promise(resolve => canvas.toBlob(resolve, 'image/png'));
}

function setupMorphCanvas(width, height) {
	morphCanvas.width = width;
	morphCanvas.height = height;
	//processedImage.parentNode.insertBefore(morphCanvas, processedImage);
	processedImage.style.display = "none";
}

async function morphImages(oldImg, newImg, duration = 300) {
	if (!oldImg) {
		morphCanvas.style.display = 'none';
		processedImage.style.display = 'block';
		processedImage.src = newImg.src;
		oldImageData = new Image();
		oldImageData.src = newImg.src;
		return;
	}

	if (morphCanvas.width !== newImg.width || morphCanvas.height !== newImg.height) {
		setupMorphCanvas(512, 512);
	}

	morphCanvas.style.display = 'block';
	//processedImage.style.display = 'none';

	return new Promise(resolve => {
		let start = null;

		function animate(timestamp) {
			if (!start) start = timestamp;
			let progress = (timestamp - start) / duration;
			if (progress > 1) progress = 1;

			morphCtx.clearRect(0, 0, morphCanvas.width, morphCanvas.height);
			morphCtx.globalAlpha = 1;
			morphCtx.drawImage(oldImg, 0, 0);
			morphCtx.globalAlpha = progress;
			morphCtx.drawImage(newImg, 0, 0);

			if (progress < 1) {
				requestAnimationFrame(animate);
			} else {
				const newProcessedImg = new Image();
				newProcessedImg.onload = () => {
					morphCanvas.style.display = 'none';
					processedImage.style.display = 'block';
					processedImage.src = newProcessedImg.src;
					morphCtx.clearRect(0, 0, morphCanvas.width, morphCanvas.height);
					oldImageData = newProcessedImg;
					resolve();
				};
				newProcessedImg.src = newImg.src;
			}
		}
		requestAnimationFrame(animate);
	});
}

function get_avg_latency () {
	if(avg_latency.length == 0) {
		return 1000;
	}

	let sum = 0;
	for(let i = 0; i < avg_latency.length; i++) {
		sum += avg_latency[i];
	}
	return sum / avg_latency.length;
}

async function sendImage() {
	const blob = await getFrameBlob(video);
	if (!blob) return showError("Konnte Bildblob nicht erstellen");

	const form = new FormData();
	form.append("strength", document.getElementById('strength').value);
	form.append("input", blob, "frame.png");
	form.append("prompt", promptInput.value || "");
	form.append("negative_prompt", document.getElementById('negative_prompt').value || "");
	form.append("steps", document.getElementById('num_inference_steps').value);
	form.append("scale", document.getElementById('guidance_scale').value);
	form.append("model", document.getElementById('model_select').value);
	form.append("seed", "33");

	const start = performance.now();

	try {
		const res = await fetch("/generate", { method: "POST", body: form });
		if (!res.ok) throw new Error("Server antwortete mit Status " + res.status);

		const blob = await res.blob();
		const objectUrl = URL.createObjectURL(blob);

		const newImg = new Image();
		newImg.onload = async () => {
			URL.revokeObjectURL(objectUrl);
			await morphImages(oldImageData, newImg, Math.max(200, 0.9 * Math.floor(get_avg_latency())));
			const latency = (performance.now() - start) / 1000;
			avg_latency.push(latency);
			latencyDisplay.textContent = `Verarbeitung: ${latency.toFixed(2)} Sekunden`;
			errorBox.style.display = "none";
		};
		newImg.src = objectUrl;

	} catch (e) {
		showError("Fehler beim Senden: " + e.message);
	}
}

function showError(msg) {
	errorBox.textContent = msg;
	errorBox.style.display = "block";
}

function showSpinner(text = "Bitte warten...") {
	// Container erstellen
	const container = document.createElement("div");
	container.id = "spinner-container";
	container.style.cssText = `
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0, 0, 0, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    z-index: 9999;
    font-family: sans-serif;
    color: white;
  `;

	// Spinner erstellen (CSS-Kreis)
	const spinner = document.createElement("div");
	spinner.style.cssText = `
    border: 8px solid #f3f3f3;
    border-top: 8px solid #3498db;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    animation: spin 1s linear infinite;
    margin-bottom: 15px;
  `;

	// Text
	const label = document.createElement("div");
	label.textContent = text;

	// Alles zusammenfügen
	container.appendChild(spinner);
	container.appendChild(label);
	document.body.appendChild(container);

	// Animation einfügen
	const style = document.createElement("style");
	style.textContent = `
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  `;
	document.head.appendChild(style);
}

function hideSpinner() {
	const container = document.getElementById("spinner-container");
	if (container) container.remove();
}

function sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms));
}

async function loop() {
	while (1) {
		sendImage();
		await sleep(Math.min(100, Math.max(200, get_avg_latency() / nr_gpus)));
	}
}

startWebcam().then(loop);
