const log = console.log;

async function getNumberOfGPUsUntilSuccess(retryDelay = 500) {
	showSpinner("Waiting for HPC-Server...");

	async function tryFetch() {
		try {
			const res = await fetch("/serverinfo");
			if (!res.ok) throw new Error("Antwort nicht OK");
			const data = await res.json();

			if ("available_gpus" in data) {
				console.log("Anzahl GPUs:", data.available_gpus);
				hideSpinner();
				return data.available_gpus;
			} else {
				throw new Error("available_gpus fehlt im JSON");
			}
		} catch (err) {
			console.warn("Fehler beim Abrufen, versuche erneut in", retryDelay, "ms:", err.message);
			await new Promise(resolve => setTimeout(resolve, retryDelay));
			return tryFetch();  // Wiederholung
		}
	}

	return tryFetch();
}

const morphCanvas = document.createElement('canvas');
const morphCtx = morphCanvas.getContext('2d');
const processedImage = document.getElementById("processedImage");
const video = document.getElementById('webcam');
const promptInput = document.getElementById('prompt');
const latencyDisplay = document.getElementById('latency');
const errorBox = document.getElementById('error');
var is_switching_models = false;
var nr_gpus = 0;

let oldImageData = null;
let delay = 1000;
let avg_latency = [];

function switchModels() {
	showSpinner("Switching model");
}

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

			hideSpinner();
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


let runningJobs = 0;
let lastJobId = 0;
let lastHandledJobId = 0;

const jobDurations = [];
const maxDurationsStored = 20;

function updateAvgDuration(duration) {
	jobDurations.push(duration);
	if (jobDurations.length > maxDurationsStored) jobDurations.shift();
}

function getAvgDuration() {
	if (jobDurations.length === 0) return 1000;
	return jobDurations.reduce((a,b) => a+b, 0) / jobDurations.length;
}

let lastStartTime = 0;
let targetInterval = 0; // ms zwischen Job-Starts, wird adaptiv angepasst

async function startJob() {
	if (runningJobs >= nr_gpus) return false;

	runningJobs++;
	const thisJobId = ++lastJobId;
	const startTime = performance.now();

	try {
		await sendImage(thisJobId);
		const duration = performance.now() - startTime;

		updateAvgDuration(duration);

		if (thisJobId >= lastHandledJobId) {
			lastHandledJobId = thisJobId;
		}
	} catch(e) {
		console.error('Job failed', e);
	} finally {
		runningJobs--;
	}
	return true;
}

async function loop() {
	nr_gpus = await getNumberOfGPUsUntilSuccess();

	targetInterval = getAvgDuration() / nr_gpus;

	while(true) {
		const now = performance.now();
		const sinceLastStart = now - lastStartTime;

		// Job starten wenn:
		// 1) Noch Slots frei sind
		// 2) Mindestintervall seit letztem Start vergangen ist
		if (runningJobs < nr_gpus && sinceLastStart >= targetInterval) {
			lastStartTime = now;
			const started = startJob();

			// Nach Jobstart direkt neu TargetInterval berechnen
			const avgDur = getAvgDuration();
			// Ziel: jobs starten mit Abstand avgDur/nr_gpus ± Toleranz
			targetInterval = avgDur / nr_gpus;

			// Optional: Interval auf vernünftige Grenzen begrenzen, z.B. 30ms - 100ms
			targetInterval = Math.min(Math.max(targetInterval, 30), 100);
		} else {
			// Sonst kurz warten und dann nochmal prüfen
			await sleep(10);
		}
	}
}

startWebcam().then(loop);
