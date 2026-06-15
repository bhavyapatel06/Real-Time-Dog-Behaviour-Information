let model;
const CLASS_NAMES = ['happy', 'angry', 'alert', 'relax'];
// --- BROWSER BYPASS FOR KERAS AUGMENTATION LAYERS ---
// This tells the browser to safely ignore training-only layers

class RandomFlip extends tf.layers.Layer {
    constructor(config) { super(config); }
    computeOutputShape(inputShape) { return inputShape; }
    call(inputs) { return inputs; } // Just pass the image through untouched
    static get className() { return 'RandomFlip'; }
}
tf.serialization.registerClass(RandomFlip);

class RandomRotation extends tf.layers.Layer {
    constructor(config) { super(config); }
    computeOutputShape(inputShape) { return inputShape; }
    call(inputs) { return inputs; }
    static get className() { return 'RandomRotation'; }
}
tf.serialization.registerClass(RandomRotation);

class RandomZoom extends tf.layers.Layer {
    constructor(config) { super(config); }
    computeOutputShape(inputShape) { return inputShape; }
    call(inputs) { return inputs; }
    static get className() { return 'RandomZoom'; }
}
tf.serialization.registerClass(RandomZoom);
// -----------------------------------------------------

const videoElement = document.getElementById('webcam');
const startBtn = document.getElementById('start-btn');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');

// Async initialization sequence
async function loadModel() {
    try {
        console.log("Attempting to load model...");
        
        // We load the model, and then ensure it has the correct input shape
        model = await tf.loadLayersModel('./model.json');
        
        // This 'dummy' prediction initializes the layers so the error goes away
        const dummyInput = tf.zeros([1, 224, 224, 3]);
        model.predict(dummyInput);
        
        document.getElementById('status-text').innerText = "System ready. Initialize scanner.";
        document.getElementById('start-btn').disabled = false;
        console.log("Model initialized successfully!");
    } catch (err) {
        console.error("Critical Load Failure:", err);
        document.getElementById('status-text').innerText = "Load Error: Check console.";
    }
}


// Adaptive multi-device camera configuration initialization
async function setupCamera() {
    startBtn.style.display = 'none';
    videoElement.style.display = 'block';
    statusText.innerText = "Configuring media capture hardware streams...";

    // Strategy 1: Attempt mobile optimal landscape rear environment camera configuration
    let constraints = {
        video: { facingMode: 'environment', width: 224, height: 224 },
        audio: false
    };

    try {
        await startMediaStream(constraints);
    } catch (mobileError) {
        console.warn("Rear environment camera not found. Executing standard fallback protocol...");
        
        // Strategy 2: Fallback to whatever camera exists (e.g. standard laptop webcam configuration)
        constraints = {
            video: true,
            audio: false
        };
        
        try {
            await startMediaStream(constraints);
        } catch (laptopError) {
            statusText.innerText = "Capture initialization failure: Hardware unavailable.";
            statusDot.className = "status-indicator error";
            console.error(laptopError);
        }
    }
}

// Secondary execution engine for video bindings
async function startMediaStream(config) {
    const stream = await navigator.mediaDevices.getUserMedia(config);
    videoElement.srcObject = stream;
    
    videoElement.onloadedmetadata = () => {
        statusText.innerText = "Live processing pipeline telemetry active";
        statusDot.className = "status-indicator active";
        predictFrame();
    };
}

// Processing cycle looping framework
async function predictFrame() {
    // ─── SAFETY CHECK ──────────────────────────────────────────────────
    // If the webcam is ready but the model is still downloading,
    // this keeps the loop going safely without crashing the app!
    if (!model) {
        console.log("Waiting for AI model to finish loading...");
        requestAnimationFrame(predictFrame);
        return;
    }
    // ───────────────────────────────────────────────────────────────────

    if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        const predictions = tf.tidy(() => {
            const img = tf.browser.fromPixels(videoElement);
            const resized = tf.image.resizeBilinear(img, [224, 224]);
            const normalized = resized.div(tf.scalar(127.5)).sub(tf.scalar(1));
            const batched = normalized.expandDims(0);
            
            return model.predict(batched).dataSync();
        });

        updateDashboard(predictions);
    }
    requestAnimationFrame(predictFrame);
}

// Interface display updater 
function updateDashboard(scores) {
    const targetEmotions = {
        happy: scores[0],
        angry: scores[1],
        alert: scores[2],
        relax: scores[3]
    };

    // Determine the current absolute maximum value for active state styling updates
    let maxScore = -1;
    let winningEmotion = "";
    
    CLASS_NAMES.forEach((emotion, index) => {
        const currentScore = scores[index];
        if (currentScore > maxScore) {
            maxScore = currentScore;
            winningEmotion = emotion;
        }
    });

    // Update each visual metric row block completely
    CLASS_NAMES.forEach((emotion) => {
        const pctValue = (targetEmotions[emotion] * 100).toFixed(1);
        
        // Target specific nodes
        const barNode = document.getElementById(`bar-${emotion}`);
        const textNode = document.getElementById(`pct-happy`);
        const cardNode = document.getElementById(`card-${emotion}`);
        
        // Correct individual target mapping updates
        document.getElementById(`bar-${emotion}`).style.width = `${pctValue}%`;
        document.getElementById(`pct-${emotion}`).innerText = `${pctValue}%`;
        
        // Highlight active matrix card element systematically
        if (emotion === winningEmotion) {
            document.getElementById(`card-${emotion}`).classList.add('active-state');
        } else {
            document.getElementById(`card-${emotion}`).classList.remove('active-state');
        }
    });
}

startBtn.addEventListener('click', setupCamera);
loadModel();
