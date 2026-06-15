let model;
const CLASS_NAMES = ['happy', 'angry', 'alert', 'relax'];

const videoElement = document.getElementById('webcam');
const startBtn = document.getElementById('start-btn');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');

async function loadModel() {
    try {
        console.log("Attempting to load model...");
        // Use the path where your model.json is stored
        model = await tf.loadLayersModel('./model.json');
        
        // Warm-up prediction
        const dummyInput = tf.zeros([1, 224, 224, 3]);
        model.predict(dummyInput);
        
        statusText.innerText = "System ready. Initialize scanner.";
        startBtn.disabled = false;
        console.log("Model initialized successfully!");
    } catch (err) {
        console.error("Critical Load Failure:", err);
        statusText.innerText = "Load Error: Check console.";
    }
}

async function setupCamera() {
    startBtn.style.display = 'none';
    videoElement.style.display = 'block';
    statusText.innerText = "Configuring hardware...";

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: 224, height: 224 },
            audio: false
        });
        videoElement.srcObject = stream;
        videoElement.onloadedmetadata = () => {
            statusText.innerText = "Live processing active";
            statusDot.className = "status-indicator active";
            predictFrame();
        };
    } catch (err) {
        console.error("Camera Error:", err);
        statusText.innerText = "Camera access denied.";
    }
}

async function predictFrame() {
    if (!model) {
        requestAnimationFrame(predictFrame);
        return;
    }

    if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        tf.tidy(() => {
            const img = tf.browser.fromPixels(videoElement);
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            const normalized = resized.div(tf.scalar(127.5)).sub(tf.scalar(1));
            const batched = normalized.expandDims(0);
            const prediction = model.predict(batched);
            updateDashboard(prediction.dataSync());
        });
    }
    requestAnimationFrame(predictFrame);
}

function updateDashboard(scores) {
    let maxScore = -1;
    let winningEmotion = "";
    
    CLASS_NAMES.forEach((emotion, index) => {
        if (scores[index] > maxScore) {
            maxScore = scores[index];
            winningEmotion = emotion;
        }
    });

    CLASS_NAMES.forEach((emotion, index) => {
        const pctValue = (scores[index] * 100).toFixed(1);
        const bar = document.getElementById(`bar-${emotion}`);
        const text = document.getElementById(`pct-${emotion}`);
        const card = document.getElementById(`card-${emotion}`);
        
        if(bar) bar.style.width = `${pctValue}%`;
        if(text) text.innerText = `${pctValue}%`;
        if(card) {
            if (emotion === winningEmotion) card.classList.add('active-state');
            else card.classList.remove('active-state');
        }
    });
}

startBtn.addEventListener('click', setupCamera);
loadModel();
